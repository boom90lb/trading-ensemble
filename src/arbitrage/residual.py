"""Causal OU s-score signals on PCA factor residuals (Avellaneda-Lee).

Per Avellaneda & Lee (2010): at each bar, every active stock's trailing returns
are regressed on the current eigenportfolio factor returns; the cumulative
residual is fit as an AR(1)/discrete OU process; the deviation of the current
cumulative residual from its drift-adjusted equilibrium, in equilibrium-sigma
units, is the s-score. A per-stock state machine converts s-scores into
long/flat/short units, and the book builder nets stock legs plus eigenportfolio
hedge legs into one per-symbol close-time target row — the exact format
``src.execution.target_weights.backtest_target_weights`` consumes.

Causality invariant: the s-score, betas, and eigenportfolios attributed to bar
``t`` use bars ``<= t`` only, so appending future bars never changes them.
Invalid OU fits (non-mean-reverting, degenerate variance) and half-life
violations produce *no* signal — they are counted, because a high invalid rate
is itself a headline diagnostic of the factor model's health.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.arbitrage.factors import (
    ResidualStatArbConfig,
    batched_factor_ols,
    compute_eligibility,
    compute_returns,
    estimate_eigenportfolios,
    etf_factor_portfolios,
    volume_time_weights,
)
from src.portfolio.construct import (
    apply_no_trade_band,
    build_residual_book_row as build_book_row,
    cap_book,
    cost_aware_band,
)

_EPS = 1e-12
# Below this equilibrium sigma the s-score denominator is noise, not signal.
_SIGMA_EQ_FLOOR = 1e-9


@dataclass(frozen=True)
class OUFit:
    """Vectorized AR(1)/OU estimates for one cross-section of residual windows."""

    b: np.ndarray
    m: np.ndarray
    sigma_eq: np.ndarray
    half_life_bars: np.ndarray
    x_end: np.ndarray
    valid: np.ndarray


def fit_ou_batch(eps_window: np.ndarray) -> OUFit:
    """Fit ``X_{n+1} = a + b X_n + zeta`` on cumulative residuals, per column.

    ``eps_window`` is (window x n_stocks) of OLS residual *returns*; the OU
    state is ``X = cumsum(eps)``. Mapping: ``m = a / (1 - b)``,
    ``sigma_eq = sqrt(Var(zeta) / (1 - b^2))``, ``half_life = -ln(2)/ln(b)``.
    A column is valid only when ``b`` is inside (0, 1) with usable distance
    from 1 and the equilibrium sigma is finite and above floor — anything else
    is "no trade", never "trade anyway".
    """
    eps = np.asarray(eps_window, dtype=float)
    if eps.ndim != 2 or eps.shape[0] < 5:
        raise ValueError(f"eps_window must be 2-D with >= 5 rows, got shape {eps.shape}")

    X = np.cumsum(eps, axis=0)
    lagged, lead = X[:-1], X[1:]
    n = lagged.shape[0]
    mean_lag = lagged.mean(axis=0)
    mean_lead = lead.mean(axis=0)
    centered_lag = lagged - mean_lag
    var_lag = (centered_lag**2).sum(axis=0)
    safe_var_lag = np.where(var_lag > _EPS, var_lag, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        b = ((centered_lag * (lead - mean_lead)).sum(axis=0)) / safe_var_lag
        a = mean_lead - b * mean_lag
        zeta = lead - a[None, :] - b[None, :] * lagged
        var_zeta = (zeta**2).sum(axis=0) / max(n - 2, 1)
        one_minus_b_sq = 1.0 - b**2
        m = a / (1.0 - b)
        sigma_eq = np.sqrt(var_zeta / np.where(one_minus_b_sq > _EPS, one_minus_b_sq, np.nan))
        half_life = np.where((b > 0.0) & (b < 1.0), -np.log(2.0) / np.log(np.clip(b, _EPS, 1.0 - _EPS)), np.inf)

    valid = (
        np.isfinite(b)
        & (b > 0.0)
        & (b < 1.0)
        & (one_minus_b_sq > _EPS)
        & np.isfinite(m)
        & np.isfinite(sigma_eq)
        & (sigma_eq > _SIGMA_EQ_FLOOR)
    )
    return OUFit(b=b, m=m, sigma_eq=sigma_eq, half_life_bars=half_life, x_end=X[-1], valid=valid)


def sscore_batch(fit: OUFit, alpha: np.ndarray) -> np.ndarray:
    """Drift-adjusted s-score: ``(X_end - m - alpha/kappa) / sigma_eq``.

    ``alpha`` is the per-bar OLS intercept (the stock's idiosyncratic drift);
    ``kappa`` per bar is ``-ln(b)``, so the A-L centering shift ``alpha/kappa``
    is ``alpha / (-ln b)`` with the annualization factors cancelled. Invalid
    columns return NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa_bar = -np.log(np.where(fit.valid, fit.b, np.nan))
        shift = np.asarray(alpha, dtype=float) / kappa_bar
        s = (fit.x_end - fit.m - shift) / fit.sigma_eq
    return np.where(fit.valid & np.isfinite(s), s, np.nan)


def next_states(states: np.ndarray, sscores: np.ndarray, ok: np.ndarray, config: ResidualStatArbConfig) -> np.ndarray:
    """One causal state-machine step over the cross-section.

    States are -1 (short) / 0 (flat) / +1 (long). Exits are checked first;
    a bar that exits may immediately re-open on the opposite side if the
    s-score has crossed the far entry band (a direct flip). ``ok=False``
    (ineligible, missing data, invalid or too-slow OU fit) forces flat
    unconditionally, and NaN s-scores can never satisfy an entry comparison.
    """
    s = np.asarray(sscores, dtype=float)
    state = np.asarray(states, dtype=np.int8)
    new = state.copy()
    with np.errstate(invalid="ignore"):
        exit_long = (state == 1) & (s > config.s_exit_long)
        exit_short = (state == -1) & (s < config.s_exit_short)
        new[exit_long | exit_short] = 0
        can_open = (state == 0) | exit_long | exit_short
        new[can_open & (s < config.s_entry_long)] = 1
        new[can_open & (s > config.s_entry_short)] = -1
    new[~np.asarray(ok, dtype=bool)] = 0
    return new


def run_state_machine(sscores: np.ndarray, ok: np.ndarray, config: ResidualStatArbConfig) -> np.ndarray:
    """Run the per-stock state machine over (n_bars x n_stocks) rows, starting flat."""
    s = np.asarray(sscores, dtype=float)
    ok_rows = np.asarray(ok, dtype=bool)
    if s.shape != ok_rows.shape or s.ndim != 2:
        raise ValueError(f"sscores {s.shape} and ok {ok_rows.shape} must be matching 2-D arrays")
    states = np.zeros(s.shape, dtype=np.int8)
    current = np.zeros(s.shape[1], dtype=np.int8)
    for t in range(s.shape[0]):
        current = next_states(current, s[t], ok_rows[t], config)
        states[t] = current
    return states


@dataclass(frozen=True)
class ResidualSignalPanel:
    """Causal per-day signal state for the whole panel.

    ``sscore``/``tradeable``/``active``/``half_life_bars`` are (day x symbol)
    frames (``half_life_bars`` is each active name's OU half-life in bars, NaN
    where unfit and inf where the OU fit is invalid); ``beta`` is
    (n_days, n_factors, n_symbols) with zeros for inactive entries (zeros, not
    NaN, so book netting never propagates NaN); ``q_index[t]`` points into
    ``eigenportfolios`` for the rebalance in force at bar ``t`` (-1 before the
    first one). Counter arrays are per-day so walk-forward folds can report
    their own invalid-OU rates.
    """

    index: pd.Index
    symbols: tuple[str, ...]
    sscore: pd.DataFrame
    tradeable: pd.DataFrame
    active: pd.DataFrame
    half_life_bars: pd.DataFrame
    beta: np.ndarray
    q_index: np.ndarray
    eigenportfolios: tuple[np.ndarray, ...]
    n_candidates: np.ndarray
    n_active: np.ndarray
    n_missing: np.ndarray
    n_invalid_ou: np.ndarray
    n_slow_ou: np.ndarray
    rebalance_positions: tuple[int, ...]
    eligible_at_rebalance: tuple[int, ...]
    explained_at_rebalance: tuple[float, ...]
    skipped_rebalances: int = 0
    config: ResidualStatArbConfig = field(default_factory=ResidualStatArbConfig)

    def diagnostics(self, start: int = 0, stop: int | None = None) -> dict[str, float]:
        """Signal-health counters aggregated over panel rows [start, stop)."""
        sl = slice(start, stop)
        evaluations = float(self.n_active[sl].sum())
        invalid = float(self.n_invalid_ou[sl].sum())
        slow = float(self.n_slow_ou[sl].sum())
        missing = float(self.n_missing[sl].sum())
        candidates = float(self.n_candidates[sl].sum())
        return {
            "signal_evaluations": evaluations,
            "invalid_ou_rate": invalid / evaluations if evaluations else float("nan"),
            "slow_ou_rate": slow / evaluations if evaluations else float("nan"),
            "missing_data_rate": missing / candidates if candidates else float("nan"),
        }


def compute_residual_signal_panel(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    config: ResidualStatArbConfig | None = None,
    membership_mask: pd.DataFrame | None = None,
) -> ResidualSignalPanel:
    """Walk the panel once, causally: weekly PCA (or fixed ETF factors), daily OLS, daily OU.

    PCA mode: at each rebalance bar ``t`` (every ``rebalance_every`` bars from
    the first bar with a full correlation window), eigenportfolios are
    re-estimated on returns ``(t - corr_window, t]`` over the then-eligible
    cross-section; the cross-section is frozen until the next rebalance. ETF
    mode: the factor portfolios are a single fixed selector of the sector ETFs
    and the cross-section is every stock column, re-screened for eligibility
    each bar. Each bar, every stock in the cross-section that is still eligible
    and has a fully-finite trailing ``regr_window`` of returns gets an OLS
    beta/alpha against the current factors and an OU fit on its residuals.
    Stocks whose factor-window returns go missing mid-week (halt/delisting)
    contribute zero to factor returns for those bars — a documented v1
    approximation.
    """
    config = config or ResidualStatArbConfig()
    if not closes.columns.equals(volumes.columns) or not closes.index.equals(volumes.index):
        raise ValueError("closes and volumes must share index and columns")
    if len(closes) <= config.warmup_bars:
        raise ValueError(f"panel has {len(closes)} rows; need more than warmup_bars={config.warmup_bars}")

    symbols = tuple(str(c) for c in closes.columns)
    index = closes.index
    n_days, n_symbols = len(index), len(symbols)
    if config.factor_mode == "etf":
        missing = set(config.etf_symbols) - set(symbols)
        if missing:
            raise ValueError(f"ETF factors absent from panel columns: {sorted(missing)}")
        n_stocks = sum(s not in set(config.etf_symbols) for s in symbols)
        if n_stocks < 2:
            raise ValueError(f"need >= 2 tradeable stock columns in ETF mode, got {n_stocks}")
    elif n_symbols < config.n_factors + 2:
        raise ValueError(f"need at least n_factors + 2 = {config.n_factors + 2} symbols, got {n_symbols}")

    returns = compute_returns(closes)
    eligibility = compute_eligibility(closes, volumes, config, membership_mask)
    R = returns.to_numpy(dtype=float)
    elig = eligibility.to_numpy(dtype=bool)
    # Volume "trading time" (A-L §6) weights ONLY the residual-regression
    # dependent variable; eligibility, PCA, and the systematic factor returns
    # stay on raw returns. With volume_time off, Rw is R, so the frozen path is
    # reproduced bit-for-bit.
    if config.volume_time:
        weights = volume_time_weights(volumes, config.volume_time_window, config.volume_time_clip)
        Rw = (returns * weights).to_numpy(dtype=float)
    else:
        Rw = R

    sscore = np.full((n_days, n_symbols), np.nan)
    half_life = np.full((n_days, n_symbols), np.nan)
    tradeable = np.zeros((n_days, n_symbols), dtype=bool)
    active = np.zeros((n_days, n_symbols), dtype=bool)
    beta = np.zeros((n_days, config.n_model_factors, n_symbols))
    q_index = np.full(n_days, -1, dtype=np.int32)
    n_candidates = np.zeros(n_days, dtype=np.int64)
    n_active = np.zeros(n_days, dtype=np.int64)
    n_missing = np.zeros(n_days, dtype=np.int64)
    n_invalid = np.zeros(n_days, dtype=np.int64)
    n_slow = np.zeros(n_days, dtype=np.int64)

    eigenportfolios: list[np.ndarray] = []
    rebalance_positions: list[int] = []
    eligible_counts: list[int] = []
    explained_shares: list[float] = []
    skipped = 0
    cross_section = np.zeros(n_symbols, dtype=bool)
    current_q = -1

    # ETF-factor mode has no PCA to re-estimate: the factor portfolios are a
    # single fixed selector (a unit long of each sector ETF), and the trading
    # cross-section is every stock column (sector ETFs are non-candidates via
    # eligibility). Because the selector extracts ETF returns under the same
    # ``window_R @ Q.T``, the loop body below is shared with the PCA path
    # verbatim — only the weekly re-estimation block is PCA-only.
    etf_mode = config.factor_mode == "etf"
    if etf_mode:
        etf_set = set(config.etf_symbols)
        stock_mask = np.array([s not in etf_set for s in symbols], dtype=bool)
        eigenportfolios.append(etf_factor_portfolios(symbols, config.etf_symbols))
        current_q = 0
        cross_section = stock_mask.copy()
        rebalance_positions.append(config.warmup_bars)
        eligible_counts.append(int(stock_mask.sum()))
        explained_shares.append(float("nan"))

    for t in range(config.corr_window, n_days):
        if not etf_mode and (t - config.corr_window) % config.rebalance_every == 0:
            elig_t = elig[t]
            if int(elig_t.sum()) >= config.n_factors + 2:
                window = R[t - config.corr_window + 1 : t + 1][:, elig_t]
                q_sub, explained = estimate_eigenportfolios(window, config.n_factors)
                q_full = np.zeros((config.n_factors, n_symbols))
                q_full[:, elig_t] = q_sub
                eigenportfolios.append(q_full)
                current_q = len(eigenportfolios) - 1
                cross_section = elig_t.copy()
                rebalance_positions.append(t)
                eligible_counts.append(int(elig_t.sum()))
                explained_shares.append(float(explained.sum()))
            else:
                skipped += 1

        if current_q < 0 or t < config.warmup_bars:
            continue
        q_index[t] = current_q

        candidates = cross_section & elig[t]
        n_candidates[t] = int(candidates.sum())
        if not candidates.any():
            continue

        window_R = R[t - config.regr_window + 1 : t + 1]
        finite = np.isfinite(window_R).all(axis=0)
        active_t = candidates & finite
        n_missing[t] = int((candidates & ~finite).sum())
        n_active[t] = int(active_t.sum())
        if not active_t.any():
            continue

        q_full = eigenportfolios[current_q]
        factor_window = np.nan_to_num(window_R, nan=0.0) @ q_full.T
        dependent_window = Rw[t - config.regr_window + 1 : t + 1]
        alpha, beta_t, residuals = batched_factor_ols(dependent_window[:, active_t], factor_window)
        fit = fit_ou_batch(residuals)
        scores = sscore_batch(fit, alpha)
        fast = fit.valid & (fit.half_life_bars <= config.max_half_life_bars)

        n_invalid[t] = int((~fit.valid).sum())
        n_slow[t] = int((fit.valid & ~fast).sum())
        active_idx = np.flatnonzero(active_t)
        sscore[t, active_idx] = scores
        half_life[t, active_idx] = fit.half_life_bars
        tradeable[t, active_idx] = fast & np.isfinite(scores)
        active[t, active_idx] = True
        beta[t][:, active_idx] = beta_t

    return ResidualSignalPanel(
        index=index,
        symbols=symbols,
        sscore=pd.DataFrame(sscore, index=index, columns=list(symbols)),
        tradeable=pd.DataFrame(tradeable, index=index, columns=list(symbols)),
        active=pd.DataFrame(active, index=index, columns=list(symbols)),
        half_life_bars=pd.DataFrame(half_life, index=index, columns=list(symbols)),
        beta=beta,
        q_index=q_index,
        eigenportfolios=tuple(eigenportfolios),
        n_candidates=n_candidates,
        n_active=n_active,
        n_missing=n_missing,
        n_invalid_ou=n_invalid,
        n_slow_ou=n_slow,
        rebalance_positions=tuple(rebalance_positions),
        eligible_at_rebalance=tuple(eligible_counts),
        explained_at_rebalance=tuple(explained_shares),
        skipped_rebalances=skipped,
        config=config,
    )
