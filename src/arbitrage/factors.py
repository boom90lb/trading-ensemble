"""Factor model for residual statistical arbitrage (Avellaneda & Lee, 2010).

Two factor constructions share one residual/OU pipeline:

* ``factor_mode="pca"`` (default, frozen v1): standardized-return correlation
  PCA over a trailing window, eigenportfolio weights ``Q[j, i] = v[j, i] /
  sigma_i``, factor returns ``F = R @ Q.T``, re-estimated weekly.
* ``factor_mode="etf"``: the factors are *exogenous* sector-ETF returns. Each
  stock is regressed on the panel of sector ETFs, so the "factor portfolio"
  that replicates factor ``j`` is simply a unit position in ETF ``j`` — a fixed
  selector matrix that plays the role PCA's ``Q`` plays in the default path
  (this is the multivariate ETF spec; A-L's headline ETF model uses each
  stock's single matched sector ETF, which needs a point-in-time GICS map).

In both modes a per-day batched OLS regresses every stock's trailing returns on
the shared factor-return window. The production-critical invariant is
causality: every estimate attributed to bar ``t`` may use bars ``<= t`` only.
Eligibility, PCA, and the OLS all read trailing windows that end at the bar
being evaluated, so the signal at the close of ``t`` is computable at that close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True)
class ResidualStatArbConfig:
    """Frozen v1 configuration for the residual (Avellaneda-Lee) stat-arb path.

    The defaults ARE the v1 frozen config: corr_window=252, regr_window=60,
    n_factors=15, weekly PCA re-estimation, A-L entry/exit s-score bands
    (open long s<-1.25, close long s>-0.50, open short s>1.25, close short
    s<0.75), and a 30-bar residual half-life cap. Changing any of these after
    seeing a run's results is a new trial and must be logged to the trial
    ledger, or the cross-run deflated Sharpe is meaningless.
    """

    corr_window: int = 252
    regr_window: int = 60
    n_factors: int = 15
    rebalance_every: int = 5
    min_price: float = 5.0
    min_median_dollar_volume: float = 1_000_000.0
    dollar_volume_window: int = 20
    s_entry_long: float = -1.25
    s_exit_long: float = -0.50
    s_entry_short: float = 1.25
    s_exit_short: float = 0.75
    max_half_life_bars: float = 30.0
    position_unit: float = 0.02
    volume_time: bool = False
    volume_time_window: int = 60
    volume_time_clip: float = 4.0
    factor_mode: str = "pca"
    etf_symbols: tuple[str, ...] = ()
    sizing_mode: str = "unit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "etf_symbols", tuple(str(s).upper() for s in self.etf_symbols))
        if self.corr_window < 20:
            raise ValueError(f"corr_window must be >= 20, got {self.corr_window}")
        if self.regr_window < 5:
            raise ValueError(f"regr_window must be >= 5, got {self.regr_window}")
        if self.regr_window > self.corr_window:
            raise ValueError("regr_window must be <= corr_window")
        if self.n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {self.n_factors}")
        if self.regr_window < self.n_factors + 5:
            raise ValueError(
                "regr_window must be >= n_factors + 5 so the factor OLS has residual degrees of freedom; "
                f"got regr_window={self.regr_window}, n_factors={self.n_factors}"
            )
        if self.rebalance_every < 1:
            raise ValueError(f"rebalance_every must be >= 1, got {self.rebalance_every}")
        if self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")
        if self.min_median_dollar_volume < 0:
            raise ValueError(f"min_median_dollar_volume must be >= 0, got {self.min_median_dollar_volume}")
        if self.dollar_volume_window < 1:
            raise ValueError(f"dollar_volume_window must be >= 1, got {self.dollar_volume_window}")
        if not self.s_entry_long < 0 < self.s_entry_short:
            raise ValueError("entry bands must straddle zero: s_entry_long < 0 < s_entry_short")
        if not self.s_entry_long < self.s_exit_long:
            raise ValueError("s_exit_long must be above s_entry_long (close band inside open band)")
        if not self.s_exit_short < self.s_entry_short:
            raise ValueError("s_exit_short must be below s_entry_short (close band inside open band)")
        if self.max_half_life_bars <= 0:
            raise ValueError(f"max_half_life_bars must be > 0, got {self.max_half_life_bars}")
        if self.position_unit <= 0:
            raise ValueError(f"position_unit must be > 0, got {self.position_unit}")
        if self.volume_time_window < 2:
            raise ValueError(f"volume_time_window must be >= 2, got {self.volume_time_window}")
        if self.volume_time_clip <= 1.0:
            raise ValueError(f"volume_time_clip must be > 1, got {self.volume_time_clip}")
        if self.factor_mode not in ("pca", "etf"):
            raise ValueError(f"factor_mode must be 'pca' or 'etf', got {self.factor_mode!r}")
        if self.factor_mode == "etf":
            if not self.etf_symbols:
                raise ValueError("etf_symbols must be non-empty when factor_mode='etf'")
            if len(set(self.etf_symbols)) != len(self.etf_symbols):
                raise ValueError(f"etf_symbols must be unique, got {self.etf_symbols}")
            if self.regr_window < len(self.etf_symbols) + 5:
                raise ValueError(
                    "regr_window must be >= len(etf_symbols) + 5 so the factor OLS has residual "
                    f"degrees of freedom; got regr_window={self.regr_window}, n_etf={len(self.etf_symbols)}"
                )
        elif self.etf_symbols:
            raise ValueError("etf_symbols must be empty unless factor_mode='etf'")
        if self.sizing_mode not in ("unit", "strength"):
            raise ValueError(f"sizing_mode must be 'unit' or 'strength', got {self.sizing_mode!r}")

    @property
    def warmup_bars(self) -> int:
        """Bars of history required before the first signal can exist."""
        return self.corr_window + self.regr_window

    @property
    def n_model_factors(self) -> int:
        """Factor dimension the OLS actually fits: ETF count in ETF mode, else PCA count."""
        return len(self.etf_symbols) if self.factor_mode == "etf" else self.n_factors


def compute_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns; non-positive or non-numeric prices become NaN.

    Deliberately avoids ``pct_change`` so missing prices are never forward-filled
    into synthetic zero returns.
    """
    numeric = closes.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return (numeric / numeric.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)


def consensus_trading_days(closes: pd.DataFrame, min_fraction: float = 0.6) -> pd.Series:
    """Boolean row mask keeping only genuine trading days.

    A wide panel on the *union* of every symbol's calendar inherits phantom
    dates: vendor data sometimes carries a single stray bar on an exchange
    holiday (e.g. Memorial Day, Christmas), so the union gains a date on which
    almost every name is NaN. Under the strict ``full_history`` eligibility
    rule, one such all-NaN row benches the *entire* cross-section for a full
    ``corr_window`` afterward — a single phantom bar silently kills a year of
    signals. A real session has the vast majority of continuously-listed names
    present; a phantom has ~1, so the two are trivially separable.

    The reference is the panel's peak daily presence (the number of names that
    ever trade together), so a late-IPO ramp early in the sample does not drag
    the threshold down and drop real early sessions. Returns NaN-safe booleans
    indexed like ``closes``.
    """
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")
    present = closes.apply(pd.to_numeric, errors="coerce").gt(0).sum(axis=1)
    reference = int(present.max()) if len(present) else 0
    threshold = max(1, int(np.ceil(min_fraction * reference)))
    return present >= threshold


def volume_time_weights(volumes: pd.DataFrame, window: int, clip: float) -> pd.DataFrame:
    """Avellaneda-Lee (2010) §6 eq. (20) trading-time return weights.

    A-L measure mean reversion in *trading time* rather than calendar time,
    which for EOD signals is equivalent to multiplying each bar's classical
    return by ``<dV> / dV_t`` — the ratio of the typical trailing daily share
    volume to that bar's share volume. Low-volume moves are amplified
    (weight > 1, "we believe a low-volume print will revert") and high-volume
    moves damped (weight < 1, "we are less ready to bet against a high-volume
    print"). The ratio is clipped to ``[1/clip, clip]`` so a single near-zero-
    volume bar cannot blow up a residual; non-positive or missing volume yields
    weight 1 (no information).

    Causal: both the trailing mean and the bar's own volume are known at the
    bar's close. A-L report this helps ETF-factor strategies "unequivocally"
    but gives "no significant improvement" for PCA-factor strategies — so on
    this 15-PCA path it is an honest ablation, not an expected win.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if clip <= 1.0:
        raise ValueError(f"clip must be > 1, got {clip}")
    vol = volumes.apply(pd.to_numeric, errors="coerce")
    vol = vol.where(vol > 0)
    typical = vol.rolling(window, min_periods=max(2, window // 2)).mean()
    weight = (typical / vol).replace([np.inf, -np.inf], np.nan)
    return weight.clip(lower=1.0 / clip, upper=clip).fillna(1.0)


def compute_eligibility(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    config: ResidualStatArbConfig,
    membership_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Causal per-day tradeability mask (day x symbol).

    A symbol is eligible at bar ``t`` when, using data through ``t`` only:
    a full ``corr_window`` of finite trailing returns exists (this also
    excludes late IPOs until they accrue history), the close is at least
    ``min_price``, and the trailing ``dollar_volume_window``-bar median dollar
    volume clears ``min_median_dollar_volume``.

    ``membership_mask`` (optional, day x symbol bool) is ANDed in to enforce
    point-in-time index membership: a name trades only while it was actually in
    the universe. ``None`` is a strict no-op, so the frozen-v1 path is
    reproduced bit-for-bit; a name missing from the mask columns is treated as a
    non-member (ineligible).
    """
    numeric_close = closes.apply(pd.to_numeric, errors="coerce").where(lambda f: f > 0)
    numeric_volume = volumes.apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
    returns = compute_returns(closes)

    full_history = returns.notna().rolling(config.corr_window, min_periods=1).sum() >= config.corr_window
    price_ok = numeric_close >= config.min_price
    dollar_volume = (
        (numeric_close * numeric_volume)
        .rolling(config.dollar_volume_window, min_periods=config.dollar_volume_window)
        .median()
    )
    volume_ok = dollar_volume >= config.min_median_dollar_volume
    eligible = (full_history & price_ok.fillna(False) & volume_ok.fillna(False)).astype(bool)
    # Point-in-time membership gate (optional): a name only trades while it was
    # actually in the index. None skips this entirely (frozen-v1 parity).
    if membership_mask is not None:
        member = membership_mask.reindex(
            index=eligible.index, columns=eligible.columns, fill_value=False
        ).astype(bool)
        eligible = eligible & member
    # Sector ETFs ride in the panel as factors/hedges only: they are never a
    # tradeable stock leg, so they are unconditionally ineligible as candidates.
    if config.etf_symbols:
        etf_cols = [c for c in eligible.columns if str(c) in set(config.etf_symbols)]
        if etf_cols:
            eligible[etf_cols] = False
    return eligible


def etf_factor_portfolios(symbols: tuple[str, ...], etf_symbols: tuple[str, ...]) -> np.ndarray:
    """Fixed ``(n_etf x n_symbols)`` selector: factor ``j`` is a unit long of ETF ``j``.

    In ETF-factor mode the factor return *is* a sector ETF's return, so the
    portfolio that replicates factor ``j`` is a single $1 position in ETF ``j``.
    This matrix is the ETF-mode analogue of the PCA eigenportfolio weights
    ``Q``: ``build_book_row`` consumes it identically, mapping each stock's
    factor exposure back onto the ETF columns to form the hedge legs.
    """
    col = {str(s): i for i, s in enumerate(symbols)}
    Q = np.zeros((len(etf_symbols), len(symbols)))
    for j, etf in enumerate(etf_symbols):
        if str(etf) not in col:
            raise ValueError(f"ETF factor {etf!r} is not a panel column")
        Q[j, col[str(etf)]] = 1.0
    return Q


def estimate_eigenportfolios(returns_window: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``n_factors`` eigenportfolios of a (window x n_assets) return matrix.

    Returns ``(Q, explained)`` where ``Q[j, i] = v[j, i] / sigma_i`` (the A-L
    eigenportfolio dollar weights, so factor returns are ``F = R @ Q.T``) and
    ``explained[j]`` is eigenportfolio ``j``'s share of total correlation
    variance. Eigenvectors are sign-fixed to a positive weight sum so the
    portfolios are stable across re-estimation dates.

    The window must be fully finite (eligibility guarantees this upstream).
    Zero-variance columns get zero weight in every eigenportfolio rather than
    poisoning the correlation matrix.
    """
    R = np.asarray(returns_window, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"returns_window must be 2-D, got shape {R.shape}")
    n_obs, n_assets = R.shape
    if not 1 <= n_factors <= n_assets:
        raise ValueError(f"n_factors must be in [1, n_assets={n_assets}], got {n_factors}")
    if n_obs < 3:
        raise ValueError(f"returns_window needs >= 3 rows, got {n_obs}")
    if not np.isfinite(R).all():
        raise ValueError("returns_window must be fully finite; filter eligibility first")

    sigma = R.std(axis=0, ddof=1)
    degenerate = ~np.isfinite(sigma) | (sigma <= _EPS)
    sigma_used = np.where(degenerate, 1.0, sigma)
    standardized = (R - R.mean(axis=0)) / sigma_used
    standardized[:, degenerate] = 0.0

    corr = standardized.T @ standardized / (n_obs - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1][:n_factors]
    top_values = eigenvalues[order]
    top_vectors = eigenvectors[:, order].T  # (n_factors, n_assets)

    for j in range(top_vectors.shape[0]):
        v = top_vectors[j]
        weight_sum = float(v.sum())
        flip = weight_sum < 0 or (weight_sum == 0 and v[int(np.argmax(np.abs(v)))] < 0)
        if flip:
            top_vectors[j] = -v

    Q = top_vectors / sigma_used
    Q[:, degenerate] = 0.0
    total = float(eigenvalues.sum())
    explained = top_values / total if total > _EPS else np.zeros_like(top_values)
    return Q, explained


def batched_factor_ols(
    returns_window: np.ndarray, factor_window: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS of every stock's return window on the shared factor-return window.

    ``returns_window`` is (window x n_stocks), ``factor_window`` is
    (window x n_factors); both must be fully finite. Returns
    ``(alpha (n_stocks,), beta (n_factors, n_stocks), residuals (window x n_stocks))``.
    The design matrix is shared across stocks, so this is one ``lstsq`` for the
    whole cross-section.
    """
    R = np.asarray(returns_window, dtype=float)
    F = np.asarray(factor_window, dtype=float)
    if R.ndim != 2 or F.ndim != 2 or R.shape[0] != F.shape[0]:
        raise ValueError(f"window mismatch: returns {R.shape} vs factors {F.shape}")
    if R.shape[0] < F.shape[1] + 2:
        raise ValueError("need more observations than factors + intercept for residual df")
    if not (np.isfinite(R).all() and np.isfinite(F).all()):
        raise ValueError("returns_window and factor_window must be fully finite")

    design = np.column_stack([np.ones(F.shape[0]), F])
    coef, *_ = np.linalg.lstsq(design, R, rcond=None)
    alpha = coef[0]
    beta = coef[1:]
    residuals = R - design @ coef
    return alpha, beta, residuals
