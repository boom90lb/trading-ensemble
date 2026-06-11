"""Rolling walk-forward evaluation for pair-based statistical arbitrage.

The key invariant is that each fold has a strict formation/test boundary:
pair discovery sees only the formation window, while spread targets are emitted
only on the following test window. Carry rules are intentionally absent; every
fold is flattened before the next fold can trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller  # type: ignore

from src.arbitrage.pairs import (
    PairCandidate,
    PairSelectionConfig,
    PairSelectionReport,
    PairSignalConfig,
    estimate_half_life,
    generate_pair_positions,
    scan_cointegrated_pairs_with_report,
)
from src.arbitrage.portfolio import (
    PortfolioBacktestResult,
    backtest_target_weights,
    combine_pair_positions,
    compute_portfolio_metrics,
)
from src.config import ExecutionConfig
from src.validation.metrics import deflated_sharpe_ratio, periodic_sharpe


@dataclass(frozen=True)
class StatArbWalkForwardConfig:
    """Controls rolling formation/test folds for stat-arb evaluation."""

    formation_bars: int = 504
    test_bars: int = 63
    step_bars: int | None = None
    min_test_bars: int = 20
    max_gross: float = 1.0
    max_symbol_abs_weight: float = 0.35
    close_positions_at_fold_end: bool = True

    def __post_init__(self) -> None:
        if self.formation_bars < 30:
            raise ValueError(f"formation_bars must be >= 30, got {self.formation_bars}")
        if self.test_bars < 2:
            raise ValueError(f"test_bars must be >= 2, got {self.test_bars}")
        if not 1 <= self.min_test_bars <= self.test_bars:
            raise ValueError("min_test_bars must be in [1, test_bars]")
        if self.step_bars is not None and self.step_bars < self.test_bars:
            raise ValueError("step_bars must be >= test_bars so test windows do not overlap")
        if self.max_gross <= 0:
            raise ValueError(f"max_gross must be > 0, got {self.max_gross}")
        if self.max_symbol_abs_weight <= 0:
            raise ValueError(
                f"max_symbol_abs_weight must be > 0, got {self.max_symbol_abs_weight}"
            )
        if not self.close_positions_at_fold_end:
            raise ValueError("carry rules are not implemented; folds must be flattened")

    @property
    def effective_step_bars(self) -> int:
        return self.step_bars if self.step_bars is not None else self.test_bars


@dataclass(frozen=True)
class StatArbFoldResult:
    """Diagnostics and metrics for one walk-forward fold."""

    fold: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    formation_rows: int
    test_rows: int
    selected_pairs: tuple[PairCandidate, ...]
    selection_report: PairSelectionReport
    added_pairs: tuple[str, ...]
    removed_pairs: tuple[str, ...]
    pair_turnover: int
    pair_turnover_rate: float
    test_pair_diagnostics: tuple[dict[str, float | int | str | None], ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class StatArbWalkForwardResult:
    """End-to-end walk-forward stat-arb result."""

    folds: tuple[StatArbFoldResult, ...]
    portfolio: PortfolioBacktestResult
    pair_trial_sharpes: tuple[float, ...]
    summary: dict[str, float]


@dataclass(frozen=True)
class _FoldSlices:
    fold: int
    formation: slice
    test: slice
    next_open_position: int


def iter_walk_forward_slices(n_obs: int, config: StatArbWalkForwardConfig) -> Iterator[_FoldSlices]:
    """Yield integer formation/test slices with one post-test open reserved."""
    if n_obs < config.formation_bars + config.min_test_bars + 1:
        raise ValueError(
            "Need at least formation_bars + min_test_bars + 1 rows; "
            f"got {n_obs}, formation_bars={config.formation_bars}, "
            f"min_test_bars={config.min_test_bars}"
        )

    fold = 0
    formation_start = 0
    last_test_end = n_obs - 1
    while True:
        formation_end = formation_start + config.formation_bars
        test_start = formation_end
        if test_start + config.min_test_bars > last_test_end:
            break
        test_end = min(test_start + config.test_bars, last_test_end)
        yield _FoldSlices(
            fold=fold,
            formation=slice(formation_start, formation_end),
            test=slice(test_start, test_end),
            next_open_position=test_end,
        )
        fold += 1
        formation_start += config.effective_step_bars


def _numeric_prices(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).sort_index()


def _pair_key(candidate: PairCandidate) -> str:
    return f"{candidate.asset_y}/{candidate.asset_x}"


def _candidate_to_dict(candidate: PairCandidate) -> dict[str, float | int | str]:
    return {
        "asset_y": candidate.asset_y,
        "asset_x": candidate.asset_x,
        "alpha": float(candidate.alpha),
        "beta": float(candidate.beta),
        "coint_pvalue": float(candidate.coint_pvalue),
        "coint_stat": float(candidate.coint_stat),
        "adf_pvalue": float(candidate.adf_pvalue),
        "adf_stat": float(candidate.adf_stat),
        "half_life": float(candidate.half_life),
        "beta_drift": float(candidate.beta_drift),
        "return_corr": float(candidate.return_corr),
        "n_obs": int(candidate.n_obs),
    }


def pair_selection_report_to_dict(report: PairSelectionReport) -> dict[str, object]:
    """Convert a pair-selection report to a deterministic JSON-compatible dict."""
    return {
        "n_symbols": int(report.n_symbols),
        "n_symbol_pairs": int(report.n_symbol_pairs),
        "n_raw_candidates": int(report.n_raw_candidates),
        "fdr_alpha": float(report.fdr_alpha),
        "fdr_cutoff": None if report.fdr_cutoff is None else float(report.fdr_cutoff),
        "rejection_counts": dict(report.rejection_counts),
        "raw_candidates": [_candidate_to_dict(c) for c in report.raw_candidates],
        "selected_candidates": [_candidate_to_dict(c) for c in report.selected_candidates],
    }


def _series_stat(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _test_pair_diagnostics(
    close_prices: pd.DataFrame, candidate: PairCandidate
) -> dict[str, float | int | str | None]:
    pair = close_prices[[candidate.asset_y, candidate.asset_x]].apply(pd.to_numeric, errors="coerce")
    pair = pair.where(pair > 0).dropna()
    out: dict[str, float | int | str | None] = {
        "pair": _pair_key(candidate),
        "n_obs": int(len(pair)),
        "return_corr": None,
        "spread_std": None,
        "adf_pvalue": None,
        "adf_stat": None,
        "half_life": None,
    }
    if len(pair) < 3:
        return out

    log_pair = np.log(pair)
    spread = log_pair[candidate.asset_y] - candidate.alpha - candidate.beta * log_pair[candidate.asset_x]
    spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
    returns = log_pair.diff().dropna()
    if len(returns) >= 2:
        out["return_corr"] = _series_stat(float(returns.corr().iloc[0, 1]))
    if len(spread) >= 3:
        out["spread_std"] = _series_stat(float(spread.std(ddof=1)))
        out["half_life"] = _series_stat(estimate_half_life(spread))
        try:
            adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="aic")
            out["adf_stat"] = _series_stat(float(adf_stat))
            out["adf_pvalue"] = _series_stat(float(adf_pvalue))
        except Exception:
            pass
    return out


def _force_fold_flat(targets: pd.DataFrame) -> pd.DataFrame:
    """Zero close-time targets early enough that no exposure crosses the fold end."""
    out = targets.copy()
    if out.empty:
        return out
    tail_rows = min(2, len(out))
    out.iloc[-tail_rows:] = 0.0
    return out


def _empty_targets(index: pd.Index, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=symbols)


def _slice_portfolio_result(
    result: PortfolioBacktestResult,
    eval_index: pd.Index,
) -> PortfolioBacktestResult:
    returns_index = result.returns.index.intersection(eval_index)
    returns = result.returns.loc[returns_index]
    costs = result.costs.loc[returns_index]
    equity = (1.0 + returns).cumprod().rename("equity")
    target_weights = result.target_weights.loc[result.target_weights.index.intersection(eval_index)]
    fill_weights = result.fill_weights.loc[result.fill_weights.index.intersection(eval_index)]
    return PortfolioBacktestResult(
        returns=returns,
        equity=equity,
        target_weights=target_weights,
        fill_weights=fill_weights,
        costs=costs,
        metrics=compute_portfolio_metrics(returns, equity, costs),
    )


def _fold_metrics_from_result(
    result: PortfolioBacktestResult,
    test_index: pd.Index,
) -> dict[str, float]:
    returns_index = result.returns.index.intersection(test_index)
    returns = result.returns.loc[returns_index]
    costs = result.costs.loc[returns_index]
    equity = (1.0 + returns).cumprod().rename("equity")
    return compute_portfolio_metrics(returns, equity, costs)


def _pair_trial_sharpe(
    open_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig,
) -> float:
    try:
        result = backtest_target_weights(open_prices, target_weights, execution=execution)
    except ValueError:
        return float("nan")
    return float(periodic_sharpe(result.returns))


def run_stat_arb_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    selection_config: PairSelectionConfig | None = None,
    signal_config: PairSignalConfig | None = None,
    walk_config: StatArbWalkForwardConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> StatArbWalkForwardResult:
    """Run rolling WFO pair selection, signal generation, and net accounting."""
    selection_config = selection_config or PairSelectionConfig()
    signal_config = signal_config or PairSignalConfig()
    walk_config = walk_config or StatArbWalkForwardConfig()
    execution = execution or ExecutionConfig()

    closes = _numeric_prices(close_prices)
    opens = _numeric_prices(open_prices)
    common_index = closes.index.intersection(opens.index)
    symbols = sorted(set(closes.columns).intersection(opens.columns))
    if len(symbols) < 2:
        raise ValueError("Need at least two symbols with both open and close prices")
    closes = closes.loc[common_index, symbols]
    opens = opens.loc[common_index, symbols]
    if len(closes) < walk_config.formation_bars + walk_config.min_test_bars + 1:
        raise ValueError("Not enough aligned rows for the requested stat-arb WFO")

    full_targets = _empty_targets(closes.index, symbols)
    fold_shells: list[dict[str, object]] = []
    previous_pairs: set[str] = set()
    all_test_index = pd.Index([])
    pair_trial_sharpes: list[float] = []

    for slices in iter_walk_forward_slices(len(closes), walk_config):
        formation_close = closes.iloc[slices.formation]
        test_close = closes.iloc[slices.test]
        test_index = test_close.index
        signal_close = closes.iloc[slices.formation.start:slices.test.stop]
        report = scan_cointegrated_pairs_with_report(formation_close, selection_config)
        candidates = list(report.selected_candidates)

        pair_targets: list[pd.DataFrame] = []
        local_open = opens.iloc[slices.test.start:slices.next_open_position + 1]
        for candidate in candidates:
            signal_frame = generate_pair_positions(signal_close, candidate, signal_config)
            target = signal_frame.target_weights.reindex(test_index).fillna(0.0)
            target = _force_fold_flat(target)
            pair_targets.append(target)

            local_pair_target = target.reindex(local_open.index).fillna(0.0)
            pair_trial_sharpes.append(_pair_trial_sharpe(local_open, local_pair_target, execution))

        if pair_targets:
            combined = combine_pair_positions(
                pair_targets,
                max_gross=walk_config.max_gross,
                max_symbol_abs_weight=walk_config.max_symbol_abs_weight,
            )
            combined = combined.reindex(test_index).reindex(columns=symbols, fill_value=0.0).fillna(0.0)
            combined = _force_fold_flat(combined)
        else:
            combined = _empty_targets(test_index, symbols)

        full_targets.loc[test_index, symbols] = combined
        current_pairs = {_pair_key(candidate) for candidate in candidates}
        added = tuple(sorted(current_pairs - previous_pairs))
        removed = tuple(sorted(previous_pairs - current_pairs))
        union_count = len(current_pairs | previous_pairs)
        turnover = len(added) + len(removed)
        turnover_rate = float(turnover / union_count) if union_count else 0.0
        previous_pairs = current_pairs
        all_test_index = all_test_index.union(test_index)

        fold_shells.append(
            {
                "slices": slices,
                "report": report,
                "candidates": tuple(candidates),
                "added": added,
                "removed": removed,
                "pair_turnover": turnover,
                "pair_turnover_rate": turnover_rate,
                "test_pair_diagnostics": tuple(
                    _test_pair_diagnostics(test_close, candidate) for candidate in candidates
                ),
            }
        )

    if not fold_shells:
        raise ValueError("No walk-forward folds were produced")

    full_portfolio = backtest_target_weights(opens, full_targets, execution=execution)
    portfolio = _slice_portfolio_result(full_portfolio, all_test_index)

    folds: list[StatArbFoldResult] = []
    for shell in fold_shells:
        slices = shell["slices"]
        if not isinstance(slices, _FoldSlices):
            raise TypeError("internal fold slice metadata corrupted")
        formation_index = closes.iloc[slices.formation].index
        test_index = closes.iloc[slices.test].index
        folds.append(
            StatArbFoldResult(
                fold=slices.fold,
                formation_start=formation_index[0],
                formation_end=formation_index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                formation_rows=len(formation_index),
                test_rows=len(test_index),
                selected_pairs=shell["candidates"],  # type: ignore[arg-type]
                selection_report=shell["report"],  # type: ignore[arg-type]
                added_pairs=shell["added"],  # type: ignore[arg-type]
                removed_pairs=shell["removed"],  # type: ignore[arg-type]
                pair_turnover=int(shell["pair_turnover"]),
                pair_turnover_rate=float(shell["pair_turnover_rate"]),
                test_pair_diagnostics=shell["test_pair_diagnostics"],  # type: ignore[arg-type]
                metrics=_fold_metrics_from_result(full_portfolio, test_index),
            )
        )

    finite_trial_sharpes = np.asarray([s for s in pair_trial_sharpes if np.isfinite(s)], dtype=float)
    pair_set_dsr = (
        deflated_sharpe_ratio(portfolio.returns, finite_trial_sharpes)
        if finite_trial_sharpes.size >= 2
        else float("nan")
    )
    summary = dict(portfolio.metrics)
    summary.update(
        {
            "n_folds": float(len(folds)),
            "avg_selected_pairs": float(np.mean([len(f.selected_pairs) for f in folds])),
            "total_pair_turnover": float(sum(f.pair_turnover for f in folds)),
            "avg_pair_turnover_rate": float(np.mean([f.pair_turnover_rate for f in folds])),
            "formation_symbol_pair_tests": float(
                sum(f.selection_report.n_symbol_pairs for f in folds)
            ),
            "raw_pair_candidates": float(sum(f.selection_report.n_raw_candidates for f in folds)),
            "pair_trial_count": float(len(pair_trial_sharpes)),
            "finite_pair_trial_sharpes": float(finite_trial_sharpes.size),
            "pair_set_dsr": float(pair_set_dsr),
        }
    )
    return StatArbWalkForwardResult(
        folds=tuple(folds),
        portfolio=portfolio,
        pair_trial_sharpes=tuple(float(s) for s in pair_trial_sharpes),
        summary=summary,
    )


def fold_result_to_dict(fold: StatArbFoldResult) -> dict[str, object]:
    """Convert a fold result to a deterministic JSON-compatible dict."""
    return {
        "fold": int(fold.fold),
        "formation_start": fold.formation_start.isoformat(),
        "formation_end": fold.formation_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "formation_rows": int(fold.formation_rows),
        "test_rows": int(fold.test_rows),
        "selected_pairs": [_candidate_to_dict(c) for c in fold.selected_pairs],
        "selection_report": pair_selection_report_to_dict(fold.selection_report),
        "added_pairs": list(fold.added_pairs),
        "removed_pairs": list(fold.removed_pairs),
        "pair_turnover": int(fold.pair_turnover),
        "pair_turnover_rate": float(fold.pair_turnover_rate),
        "test_pair_diagnostics": list(fold.test_pair_diagnostics),
        "metrics": {k: float(v) for k, v in sorted(fold.metrics.items())},
    }
