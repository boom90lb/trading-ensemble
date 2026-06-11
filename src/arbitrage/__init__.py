"""Statistical-arbitrage research primitives.

This package is deliberately separate from the single-symbol forecast ensemble:
arbitrage needs cross-asset selection, hedge-aware positions, and portfolio-level
accounting rather than independent ticker forecasts.
"""

from src.arbitrage.pairs import (
    PairCandidate,
    PairSelectionConfig,
    PairSelectionReport,
    PairSignalConfig,
    PairSignalFrame,
    benjamini_hochberg_mask,
    estimate_half_life,
    generate_pair_positions,
    scan_cointegrated_pairs,
    scan_cointegrated_pairs_with_report,
)
from src.arbitrage.portfolio import (
    PortfolioBacktestResult,
    backtest_target_weights,
    combine_pair_positions,
    compute_portfolio_metrics,
)
from src.arbitrage.walk_forward import (
    StatArbFoldResult,
    StatArbWalkForwardConfig,
    StatArbWalkForwardResult,
    fold_result_to_dict,
    iter_walk_forward_slices,
    pair_selection_report_to_dict,
    run_stat_arb_walk_forward,
)

__all__ = [
    "PairCandidate",
    "PairSelectionConfig",
    "PairSelectionReport",
    "PairSignalConfig",
    "PairSignalFrame",
    "PortfolioBacktestResult",
    "StatArbFoldResult",
    "StatArbWalkForwardConfig",
    "StatArbWalkForwardResult",
    "backtest_target_weights",
    "benjamini_hochberg_mask",
    "combine_pair_positions",
    "compute_portfolio_metrics",
    "estimate_half_life",
    "fold_result_to_dict",
    "generate_pair_positions",
    "iter_walk_forward_slices",
    "pair_selection_report_to_dict",
    "run_stat_arb_walk_forward",
    "scan_cointegrated_pairs",
    "scan_cointegrated_pairs_with_report",
]
