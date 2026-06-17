"""Compatibility exports for portfolio target-weight accounting."""

from src.execution.target_weights import (
    PortfolioBacktestResult,
    backtest_target_weights,
    combine_pair_positions,
    compute_portfolio_metrics,
    metrics_to_frame,
    scale_to_max_gross,
)

__all__ = [
    "PortfolioBacktestResult",
    "backtest_target_weights",
    "combine_pair_positions",
    "compute_portfolio_metrics",
    "metrics_to_frame",
    "scale_to_max_gross",
]
