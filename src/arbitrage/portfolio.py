"""Portfolio-level accounting for stat-arb target weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.config import ExecutionConfig
from src.validation.metrics import (
    calmar_ratio,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
)


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Vectorized portfolio backtest output."""

    returns: pd.Series
    equity: pd.Series
    target_weights: pd.DataFrame
    fill_weights: pd.DataFrame
    costs: pd.DataFrame
    metrics: dict[str, float]


def _aligned_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def combine_pair_positions(
    pair_weights: Iterable[pd.DataFrame],
    max_gross: float = 1.0,
    max_symbol_abs_weight: float = 0.35,
) -> pd.DataFrame:
    """Combine pair target weights into one capped portfolio target matrix."""
    if max_gross <= 0:
        raise ValueError(f"max_gross must be > 0, got {max_gross}")
    if max_symbol_abs_weight <= 0:
        raise ValueError(f"max_symbol_abs_weight must be > 0, got {max_symbol_abs_weight}")

    frames = [w.copy() for w in pair_weights]
    if not frames:
        return pd.DataFrame()
    index = frames[0].index
    for frame in frames[1:]:
        index = index.union(frame.index)
    symbols = sorted({symbol for frame in frames for symbol in frame.columns})
    combined = pd.DataFrame(0.0, index=index, columns=symbols)
    for frame in frames:
        aligned = frame.reindex(index=index).reindex(columns=symbols, fill_value=0.0).fillna(0.0)
        combined = combined.add(aligned, fill_value=0.0)

    combined = combined.clip(lower=-max_symbol_abs_weight, upper=max_symbol_abs_weight)
    gross = combined.abs().sum(axis=1)
    scale = (max_gross / gross.where(gross > max_gross)).fillna(1.0).clip(upper=1.0)
    return combined.mul(scale, axis=0).fillna(0.0)


def _cost_rates(
    trades: pd.DataFrame,
    weights: pd.DataFrame,
    execution: ExecutionConfig,
) -> pd.DataFrame:
    trade_abs = trades.abs()
    execution_cost = trade_abs * (execution.commission_bps + execution.spread_bps) / 10_000.0
    impact_cost = trade_abs.pow(2) * execution.slippage_coeff / 10_000.0
    short_gross = weights.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_gross * execution.borrow_rate_bps_annual / 10_000.0 / execution.trading_days_per_year
    costs = pd.DataFrame(index=weights.index)
    costs["commission_spread"] = execution_cost.sum(axis=1)
    costs["impact"] = impact_cost.sum(axis=1)
    costs["borrow"] = borrow
    costs["total"] = costs.sum(axis=1)
    costs["turnover"] = trade_abs.sum(axis=1)
    costs["gross"] = weights.abs().sum(axis=1)
    costs["net"] = weights.sum(axis=1)
    return costs


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return 1.0 - equity / peak.where(peak != 0)


def compute_portfolio_metrics(returns: pd.Series, equity: pd.Series, costs: pd.DataFrame) -> dict[str, float]:
    """Compute the summary metrics used by stat-arb portfolio backtests."""
    finite = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "psr": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "avg_gross": float("nan"),
            "avg_net": float("nan"),
            "avg_turnover": float("nan"),
            "total_cost": float("nan"),
        }
    total_return = float((1.0 + finite).prod() - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / max(len(finite), 1)) - 1.0)
    annualized_vol = float(finite.std(ddof=1) * np.sqrt(252.0)) if len(finite) > 1 else float("nan")
    sr = periodic_sharpe(finite)
    sharpe = float(sr * np.sqrt(252.0)) if np.isfinite(sr) else float("nan")
    max_dd = float(_drawdown(equity).max())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "psr": probabilistic_sharpe_ratio(finite),
        "max_drawdown": max_dd,
        "calmar": calmar_ratio(annualized_return, max_dd),
        "avg_gross": float(costs["gross"].mean()),
        "avg_net": float(costs["net"].mean()),
        "avg_turnover": float(costs["turnover"].mean()),
        "total_cost": float(costs["total"].sum()),
    }


def _metrics(returns: pd.Series, equity: pd.Series, costs: pd.DataFrame) -> dict[str, float]:
    return compute_portfolio_metrics(returns, equity, costs)


def backtest_target_weights(
    open_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig | None = None,
    initial_capital: float = 1.0,
) -> PortfolioBacktestResult:
    """Backtest close-time target weights with next-open fills.

    `target_weights.loc[t]` is a decision made at close `t`, filled at open
    `t+1`, and earns open-to-open returns only after that fill.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    execution = execution or ExecutionConfig()
    prices = _aligned_numeric(open_prices)
    weights = _aligned_numeric(target_weights)
    symbols = [s for s in weights.columns if s in prices.columns]
    if not symbols:
        raise ValueError("target_weights and open_prices have no overlapping columns")

    index = prices.index.intersection(weights.index)
    prices = prices.loc[index, symbols].ffill()
    targets = weights.loc[index, symbols].fillna(0.0)
    fill_weights = targets.shift(1).fillna(0.0)
    trades = fill_weights.diff().fillna(fill_weights)
    costs = _cost_rates(trades, fill_weights, execution)

    open_to_open = prices.shift(-1) / prices - 1.0
    gross_returns = (fill_weights * open_to_open).sum(axis=1)
    returns = (gross_returns - costs["total"]).iloc[:-1].rename("daily_return")
    costs = costs.iloc[:-1]
    equity = (1.0 + returns).cumprod() * initial_capital
    if not equity.empty:
        equity.iloc[0] = initial_capital * (1.0 + returns.iloc[0])
    equity = equity.rename("equity")
    return PortfolioBacktestResult(
        returns=returns,
        equity=equity,
        target_weights=targets,
        fill_weights=fill_weights.iloc[:-1],
        costs=costs,
        metrics=_metrics(returns, equity, costs),
    )


def metrics_to_frame(metrics_by_name: Mapping[str, dict[str, float]]) -> pd.DataFrame:
    """Small reporting helper used by scripts."""
    return pd.DataFrame(metrics_by_name).T.sort_index()
