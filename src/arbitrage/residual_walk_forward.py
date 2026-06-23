"""Rolling walk-forward evaluation for residual (Avellaneda-Lee) stat-arb.

Mirrors the pairs walk-forward, with one structural difference: there is no
per-fold model selection to freeze. Eigenportfolios, betas, and OU fits re-roll
causally through test bars on trailing windows; folds exist for reporting, for
forcing positions flat at fold boundaries (carry rules are still intentionally
absent), and so the fold geometry matches the pairs path. Only hyperparameters
are fixed — and in v1 those are frozen at the config defaults.

Within-run there is exactly one trial (one frozen config), so the deflated
Sharpe for this path is computed *across runs* from the persisted trial ledger
by the CLI script, not here; the core summary exposes ``oos_periodic_sharpe``
as the ledger entry's input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.arbitrage.factors import ResidualStatArbConfig, consensus_trading_days
from src.execution.target_weights import PortfolioBacktestResult, backtest_target_weights
from src.arbitrage.residual import (
    ResidualSignalPanel,
    compute_residual_signal_panel,
    run_state_machine,
)
from src.portfolio.construct import (
    apply_no_trade_band,
    build_residual_book_row,
    cap_book,
    cost_aware_band,
    strength_multiplier,
)
from src.arbitrage.walk_forward import (
    StatArbWalkForwardConfig,
    _empty_targets,
    _fold_metrics_from_result,
    _FoldSlices,
    _force_fold_flat,
    _numeric_prices,
    _slice_portfolio_result,
    iter_walk_forward_slices,
)
from src.config import ExecutionConfig
from src.validation.metrics import periodic_sharpe

# Conviction multiplier cap for sizing_mode="strength" (plan 005): a documented
# modeling constant, NOT fitted on the backtest.
_SIZE_CAP = 2.0


@dataclass(frozen=True)
class ResidualFoldResult:
    """Diagnostics and metrics for one residual walk-forward fold."""

    fold: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    formation_rows: int
    test_rows: int
    names_traded: int
    cost_to_gross_pnl: float
    signal: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ResidualStatArbWalkForwardResult:
    """End-to-end residual walk-forward result."""

    folds: tuple[ResidualFoldResult, ...]
    portfolio: PortfolioBacktestResult
    panel: ResidualSignalPanel
    summary: dict[str, float]


def _fold_cost_share(result: PortfolioBacktestResult, test_index: pd.Index) -> float:
    """Costs as a share of gross (pre-cost) PnL magnitude over the fold.

    Values above 1 mean costs exceeded everything the signal made before
    costs — the most direct "this fold was a toll booth" diagnostic.
    """
    rows = result.returns.index.intersection(test_index)
    if rows.empty:
        return float("nan")
    costs = float(result.costs.loc[rows, "total"].sum())
    gross_pnl = float((result.returns.loc[rows] + result.costs.loc[rows, "total"]).sum())
    return costs / max(abs(gross_pnl), 1e-9)


def run_residual_stat_arb_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    signal_config: ResidualStatArbConfig | None = None,
    walk_config: StatArbWalkForwardConfig | None = None,
    execution: ExecutionConfig | None = None,
    membership_mask: pd.DataFrame | None = None,
    initial_capital: float = 1.0,
) -> ResidualStatArbWalkForwardResult:
    """Run causal residual signal generation, fold accounting, and net backtest.

    ``membership_mask`` (optional, day x symbol bool) enforces point-in-time
    index membership in eligibility; it is aligned to the (trading-day-filtered)
    panel inside ``compute_eligibility``. ``None`` reproduces frozen-v1 output.
    """
    signal_config = signal_config or ResidualStatArbConfig()
    walk_config = walk_config or StatArbWalkForwardConfig()
    execution = execution or ExecutionConfig()
    if walk_config.formation_bars < signal_config.warmup_bars:
        raise ValueError(
            "formation_bars must cover the estimator warmup: need >= corr_window + regr_window = "
            f"{signal_config.warmup_bars}, got {walk_config.formation_bars}"
        )

    closes = _numeric_prices(close_prices)
    opens = _numeric_prices(open_prices)
    vols = _numeric_prices(volumes)
    common_index = closes.index.intersection(opens.index).intersection(vols.index)
    symbols = sorted(set(closes.columns) & set(opens.columns) & set(vols.columns))
    etf_set = set(signal_config.etf_symbols)
    stock_symbols = [s for s in symbols if s not in etf_set]
    if signal_config.factor_mode == "etf":
        missing = etf_set - set(symbols)
        if missing:
            raise ValueError(f"ETF factors missing from price panels: {sorted(missing)}")
        if len(stock_symbols) < 2:
            raise ValueError(f"need >= 2 tradeable stock columns in ETF mode, got {len(stock_symbols)}")
    elif len(symbols) < signal_config.n_factors + 2:
        raise ValueError(f"need at least n_factors + 2 = {signal_config.n_factors + 2} symbols with full data")
    closes = closes.loc[common_index, symbols]
    opens = opens.loc[common_index, symbols]
    vols = vols.loc[common_index, symbols]

    # Drop phantom (non-trading) union dates before any estimator sees them: a
    # single stray holiday bar would otherwise inject a panel-wide NaN row and
    # bench the whole cross-section for a full corr_window under the strict
    # full-history eligibility rule.
    trading_days = consensus_trading_days(closes)
    n_dropped_calendar_days = int((~trading_days).sum())
    closes = closes.loc[trading_days]
    opens = opens.loc[trading_days]
    vols = vols.loc[trading_days]
    if len(closes) < walk_config.formation_bars + walk_config.min_test_bars + 1:
        raise ValueError("Not enough aligned rows for the requested residual stat-arb WFO")

    panel = compute_residual_signal_panel(closes, vols, signal_config, membership_mask)

    full_targets = _empty_targets(closes.index, symbols)
    fold_shells: list[dict[str, object]] = []
    all_test_index = pd.Index([])
    sscores = panel.sscore.to_numpy()
    tradeable = panel.tradeable.to_numpy()

    for slices in iter_walk_forward_slices(len(closes), walk_config):
        test_index = closes.index[slices.test]
        states = run_state_machine(sscores[slices.test], tradeable[slices.test], signal_config)

        rows = np.zeros((len(test_index), len(symbols)))
        for offset, t in enumerate(range(slices.test.start, slices.test.stop)):
            q_idx = int(panel.q_index[t])
            if q_idx < 0 or not states[offset].any():
                continue
            size_scale = None
            if signal_config.sizing_mode == "strength":
                # Conviction multiplier in [0, _SIZE_CAP]: 0 at the entry band, growing
                # with |s| past it (Da-Nagel-Xiu shrinkage of weak signals); cap_book
                # then redistributes the fixed gross budget toward conviction.
                entry = min(abs(signal_config.s_entry_long), abs(signal_config.s_entry_short))
                size_scale = strength_multiplier(sscores[t], entry, cap=_SIZE_CAP)
            rows[offset] = build_residual_book_row(
                states[offset],
                panel.beta[t],
                panel.eigenportfolios[q_idx],
                signal_config.position_unit,
                size_scale=size_scale,
            )
        targets = pd.DataFrame(rows, index=test_index, columns=symbols)
        targets = cap_book(targets, walk_config.max_gross, walk_config.max_symbol_abs_weight)
        if walk_config.band_mode == "cost_aware":
            # Per-name band from each name's most-recent causal OU half-life (formation
            # window only, never the test window) and the linear round-trip cost.
            per_trade_cost = (execution.commission_bps + execution.spread_bps) * 2.0 / 10_000.0
            hl_hist = panel.half_life_bars.iloc[: slices.test.start].replace([np.inf, -np.inf], np.nan)
            hl_recent = (
                hl_hist.ffill().iloc[-1].to_numpy() if len(hl_hist) else np.full(len(symbols), np.nan)
            )
            band = pd.Series(cost_aware_band(hl_recent, per_trade_cost), index=symbols)
            targets = apply_no_trade_band(targets, band)
            targets = cap_book(targets, walk_config.max_gross, walk_config.max_symbol_abs_weight)
        elif walk_config.no_trade_band > 0:
            targets = apply_no_trade_band(targets, walk_config.no_trade_band)
            targets = cap_book(targets, walk_config.max_gross, walk_config.max_symbol_abs_weight)
        targets = _force_fold_flat(targets)

        full_targets.loc[test_index, symbols] = targets
        all_test_index = all_test_index.union(test_index)
        fold_shells.append(
            {
                "slices": slices,
                "names_traded": int((targets[stock_symbols].abs() > 0).any(axis=0).sum()),
                "signal": panel.diagnostics(slices.test.start, slices.test.stop),
            }
        )

    if not fold_shells:
        raise ValueError("No walk-forward folds were produced")

    dollar_volume = (closes * vols).reindex(index=opens.index, columns=opens.columns)
    full_portfolio = backtest_target_weights(
        opens,
        full_targets,
        execution=execution,
        dollar_volume=dollar_volume,
        initial_capital=initial_capital,
    )
    portfolio = _slice_portfolio_result(full_portfolio, all_test_index)

    folds: list[ResidualFoldResult] = []
    for shell in fold_shells:
        slices = shell["slices"]
        if not isinstance(slices, _FoldSlices):
            raise TypeError("internal fold slice metadata corrupted")
        formation_index = closes.index[slices.formation]
        test_index = closes.index[slices.test]
        folds.append(
            ResidualFoldResult(
                fold=slices.fold,
                formation_start=formation_index[0],
                formation_end=formation_index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                formation_rows=len(formation_index),
                test_rows=len(test_index),
                names_traded=int(shell["names_traded"]),  # type: ignore[arg-type]
                cost_to_gross_pnl=_fold_cost_share(full_portfolio, test_index),
                signal=shell["signal"],  # type: ignore[arg-type]
                metrics=_fold_metrics_from_result(full_portfolio, test_index),
            )
        )

    evaluations = float(sum(f.signal["signal_evaluations"] for f in folds))
    weighted = {"invalid_ou_rate": float("nan"), "slow_ou_rate": float("nan")}
    if evaluations > 0:
        for key in weighted:
            weighted[key] = (
                sum(f.signal[key] * f.signal["signal_evaluations"] for f in folds if np.isfinite(f.signal[key]))
                / evaluations
            )

    summary = dict(portfolio.metrics)
    summary.update(
        {
            "n_folds": float(len(folds)),
            "n_symbols": float(len(stock_symbols)),
            "n_factor_etfs": float(len(symbols) - len(stock_symbols)),
            "n_dropped_calendar_days": float(n_dropped_calendar_days),
            "avg_names_traded": float(np.mean([f.names_traded for f in folds])),
            "signal_evaluations": evaluations,
            "invalid_ou_rate": float(weighted["invalid_ou_rate"]),
            "slow_ou_rate": float(weighted["slow_ou_rate"]),
            "n_rebalances": float(len(panel.rebalance_positions)),
            "skipped_rebalances": float(panel.skipped_rebalances),
            "avg_eligible_at_rebalance": (
                float(np.mean(panel.eligible_at_rebalance)) if panel.eligible_at_rebalance else float("nan")
            ),
            "avg_explained_variance": (
                float(np.mean(panel.explained_at_rebalance)) if panel.explained_at_rebalance else float("nan")
            ),
            "oos_periodic_sharpe": float(periodic_sharpe(portfolio.returns)),
        }
    )
    return ResidualStatArbWalkForwardResult(folds=tuple(folds), portfolio=portfolio, panel=panel, summary=summary)


def residual_fold_to_dict(fold: ResidualFoldResult) -> dict[str, object]:
    """Convert a residual fold result to a deterministic JSON-compatible dict."""
    return {
        "fold": int(fold.fold),
        "formation_start": fold.formation_start.isoformat(),
        "formation_end": fold.formation_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "formation_rows": int(fold.formation_rows),
        "test_rows": int(fold.test_rows),
        "names_traded": int(fold.names_traded),
        "cost_to_gross_pnl": float(fold.cost_to_gross_pnl),
        "signal": {k: float(v) for k, v in sorted(fold.signal.items())},
        "metrics": {k: float(v) for k, v in sorted(fold.metrics.items())},
    }
