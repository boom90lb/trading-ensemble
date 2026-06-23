"""Portfolio-level accounting for close-time target weights.

The accounting convention is deliberately the same as the order-based engine's
timing convention: a target decided from information through close ``t`` can
only fill at open ``t+1``. The simulation is sequential rather than a pure
``shift(1)`` so callers can suppress small rebalances, drop selected pending
targets at fold boundaries, and carry already-filled exposure into the next
fold.
"""

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
    """Portfolio target-weight backtest output."""

    returns: pd.Series
    equity: pd.Series
    target_weights: pd.DataFrame
    fill_weights: pd.DataFrame
    costs: pd.DataFrame
    metrics: dict[str, float]


def _aligned_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def scale_to_max_gross(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Row-scale a weight matrix so ``sum(abs(weights)) <= max_gross``."""
    if max_gross <= 0:
        raise ValueError(f"max_gross must be > 0, got {max_gross}")
    if weights.empty:
        return weights.copy()
    out = _aligned_numeric(weights).fillna(0.0)
    gross = out.abs().sum(axis=1)
    scale = (max_gross / gross.where(gross > max_gross)).fillna(1.0).clip(upper=1.0)
    return out.mul(scale, axis=0).fillna(0.0)


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
    return scale_to_max_gross(combined, max_gross=max_gross)


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return 1.0 - equity / peak.where(peak != 0)


def compute_portfolio_metrics(returns: pd.Series, equity: pd.Series, costs: pd.DataFrame) -> dict[str, float]:
    """Compute summary metrics used by target-weight portfolio backtests."""
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
    max_dd = float(_drawdown(equity).max()) if not equity.empty else float("nan")
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "psr": probabilistic_sharpe_ratio(finite),
        "max_drawdown": max_dd,
        "calmar": calmar_ratio(annualized_return, max_dd),
        "avg_gross": float(costs["gross"].mean()) if "gross" in costs else float("nan"),
        "avg_net": float(costs["net"].mean()) if "net" in costs else float("nan"),
        "avg_turnover": float(costs["turnover"].mean()) if "turnover" in costs else float("nan"),
        "total_cost": float(costs["total"].sum()) if "total" in costs else float("nan"),
    }


def _empty_costs(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "commission_spread": 0.0,
            "impact": 0.0,
            "borrow": 0.0,
            "total": 0.0,
            "turnover": 0.0,
            "gross": 0.0,
            "net": 0.0,
            "dividend_return": 0.0,
        },
        index=index,
    )


def _cost_row(
    trade: pd.Series,
    weights: pd.Series,
    execution: ExecutionConfig,
    *,
    cost_multiplier: float,
    dividend_return: float,
    dollar_volume: pd.Series | None = None,
    initial_capital: float = 1.0,
) -> dict[str, float]:
    aligned_dollar_volume = (
        None
        if dollar_volume is None
        else dollar_volume.reindex(trade.index).to_numpy(dtype=float)
    )
    return _cost_values(
        trade.to_numpy(dtype=float),
        weights.to_numpy(dtype=float),
        execution,
        cost_multiplier=cost_multiplier,
        dividend_return=dividend_return,
        dollar_volume=aligned_dollar_volume,
        initial_capital=initial_capital,
    )


def _cost_values(
    trade: np.ndarray,
    weights: np.ndarray,
    execution: ExecutionConfig,
    *,
    cost_multiplier: float,
    dividend_return: float,
    dollar_volume: np.ndarray | None = None,
    initial_capital: float = 1.0,
) -> dict[str, float]:
    trade_abs = np.abs(trade)
    commission_spread = float(
        trade_abs.sum()
        * (execution.commission_bps + execution.spread_bps)
        / 10_000.0
        * cost_multiplier
    )
    impact = float(
        np.square(trade_abs).sum()
        * execution.slippage_coeff
        / 10_000.0
        * cost_multiplier
    )
    if execution.adv_impact_coeff > 0.0 and dollar_volume is not None:
        valid = np.isfinite(dollar_volume)
        adv = np.where(valid, np.maximum(dollar_volume, execution.adv_floor_dollars), np.nan)
        participation = trade_abs * float(initial_capital) / adv
        if execution.adv_impact_model == "sqrt":
            participation = np.sqrt(participation)
        adv_bps = execution.adv_impact_coeff * participation
        impact += float(np.nansum(trade_abs * adv_bps / 10_000.0) * cost_multiplier)
    short_gross = float(np.abs(np.minimum(weights, 0.0)).sum())
    borrow = float(
        short_gross
        * execution.borrow_rate_bps_annual
        / 10_000.0
        / execution.trading_days_per_year
    )
    total = commission_spread + impact + borrow
    return {
        "commission_spread": commission_spread,
        "impact": impact,
        "borrow": borrow,
        "total": total,
        "turnover": float(trade_abs.sum()),
        "gross": float(np.abs(weights).sum()),
        "net": float(weights.sum()),
        "dividend_return": float(dividend_return),
    }


def _dividend_frame(
    dividends: pd.DataFrame | Mapping[str, pd.Series] | None,
    index: pd.Index,
    symbols: list[str],
) -> pd.DataFrame:
    if dividends is None:
        return pd.DataFrame(0.0, index=index, columns=symbols)
    if isinstance(dividends, pd.DataFrame):
        frame = dividends.copy()
    else:
        frame = pd.DataFrame({k: v for k, v in dividends.items() if v is not None})
    if frame.empty:
        return pd.DataFrame(0.0, index=index, columns=symbols)
    return (
        frame.apply(pd.to_numeric, errors="coerce")
        .reindex(index=index)
        .reindex(columns=symbols, fill_value=0.0)
        .fillna(0.0)
    )


def _target_from_row(row: pd.Series, current_weights: pd.Series) -> pd.Series | None:
    """Resolve a target row, treating all-NaN as "drop/no new pending target"."""
    if row.isna().all():
        return None
    return row.where(row.notna(), current_weights).astype(float)


def _target_from_values(row: np.ndarray, current_weights: np.ndarray) -> np.ndarray | None:
    """Resolve a target row, treating all-NaN as "drop/no new pending target"."""
    if np.isnan(row).all():
        return None
    return np.where(np.isnan(row), current_weights, row).astype(float)


def _scale_row_to_max_gross(row: np.ndarray, max_gross: float) -> np.ndarray:
    gross = float(np.abs(row).sum())
    if gross > max_gross:
        return row * (max_gross / gross)
    return row


def backtest_target_weights(
    open_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution: ExecutionConfig | None = None,
    initial_capital: float = 1.0,
    *,
    rebalance_band_weight: float = 0.0,
    rebalance_cost_multiplier: float = 1.0,
    max_gross_exposure: float | None = None,
    dividends: pd.DataFrame | Mapping[str, pd.Series] | None = None,
    dollar_volume: pd.DataFrame | None = None,
) -> PortfolioBacktestResult:
    """Backtest close-time target weights with next-open fills.

    ``target_weights.loc[t]`` is a close-time decision. It can fill only at the
    next row's open, then earns that row's open-to-open return. A target row
    whose values are all ``NaN`` is a deliberate no-op/drop marker: no pending
    order is created for the next open, while already-filled weights continue.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    if rebalance_band_weight < 0:
        raise ValueError(f"rebalance_band_weight must be >= 0, got {rebalance_band_weight}")
    if rebalance_cost_multiplier < 0:
        raise ValueError(
            f"rebalance_cost_multiplier must be >= 0, got {rebalance_cost_multiplier}"
        )
    if max_gross_exposure is not None and max_gross_exposure <= 0:
        raise ValueError(f"max_gross_exposure must be > 0, got {max_gross_exposure}")

    execution = execution or ExecutionConfig()
    prices = _aligned_numeric(open_prices)
    weights = _aligned_numeric(target_weights)
    symbols = [s for s in weights.columns if s in prices.columns]
    if not symbols:
        raise ValueError("target_weights and open_prices have no overlapping columns")

    index = prices.index.intersection(weights.index).sort_values()
    prices = prices.loc[index, symbols].ffill()
    targets = weights.loc[index, symbols]
    if max_gross_exposure is not None:
        scaled_values = scale_to_max_gross(targets.fillna(0.0), max_gross=max_gross_exposure)
        targets = scaled_values.where(targets.notna())
    divs = _dividend_frame(dividends, index, symbols)
    dvol = None if dollar_volume is None else _aligned_numeric(dollar_volume).reindex(index=index, columns=symbols)

    if len(index) < 2:
        empty = pd.Series(dtype=float, name="daily_return")
        return PortfolioBacktestResult(
            returns=empty,
            equity=pd.Series(dtype=float, name="equity"),
            target_weights=targets,
            fill_weights=pd.DataFrame(columns=symbols, dtype=float),
            costs=_empty_costs(empty.index),
            metrics=compute_portfolio_metrics(empty, pd.Series(dtype=float), _empty_costs(empty.index)),
        )

    prices_arr = prices.to_numpy(dtype=float)
    targets_arr = targets.to_numpy(dtype=float)
    divs_arr = divs.to_numpy(dtype=float)
    dvol_arr = None if dvol is None else dvol.to_numpy(dtype=float)

    current = np.zeros(len(symbols), dtype=float)
    pending: np.ndarray | None = None
    fill_rows: list[np.ndarray] = []
    return_rows: list[float] = []
    cost_rows: list[dict[str, float]] = []
    return_index = index[:-1]

    for pos, date in enumerate(return_index):
        previous = current.copy()
        if pending is not None:
            current = pending.copy()
            pending = None
        trade = current - previous

        next_pos = pos + 1
        open_ret = np.divide(
            prices_arr[next_pos],
            prices_arr[pos],
            out=np.ones(len(symbols), dtype=float),
            where=prices_arr[pos] != 0.0,
        ) - 1.0
        open_ret = np.nan_to_num(open_ret, nan=0.0, posinf=0.0, neginf=0.0)
        dividend_by_name = np.divide(
            divs_arr[next_pos],
            prices_arr[pos],
            out=np.zeros(len(symbols), dtype=float),
            where=prices_arr[pos] != 0.0,
        )
        dividend_by_name = np.nan_to_num(dividend_by_name, nan=0.0, posinf=0.0, neginf=0.0)
        dividend_return = float(
            (current * dividend_by_name).sum()
        )
        cost = _cost_values(
            trade,
            current,
            execution,
            cost_multiplier=rebalance_cost_multiplier,
            dividend_return=dividend_return,
            dollar_volume=None if dvol_arr is None else dvol_arr[pos],
            initial_capital=initial_capital,
        )
        gross_return = float((current * open_ret).sum()) + dividend_return
        return_rows.append(gross_return - cost["total"])
        fill_rows.append(current.copy())
        cost_rows.append(cost)

        row_target = _target_from_values(targets_arr[pos], current)
        if row_target is None:
            continue
        delta = row_target - current
        effective = np.where(np.abs(delta) <= rebalance_band_weight, current, row_target)
        if max_gross_exposure is not None:
            effective = _scale_row_to_max_gross(effective, max_gross_exposure)
        if not np.allclose(effective, current):
            pending = effective.astype(float)

    returns = pd.Series(return_rows, index=return_index, name="daily_return")
    fill_weights = pd.DataFrame(fill_rows, index=return_index, columns=symbols).fillna(0.0)
    costs = pd.DataFrame(cost_rows, index=return_index)
    equity = ((1.0 + returns).cumprod() * initial_capital).rename("equity")
    return PortfolioBacktestResult(
        returns=returns,
        equity=equity,
        target_weights=targets,
        fill_weights=fill_weights,
        costs=costs,
        metrics=compute_portfolio_metrics(returns, equity, costs),
    )
