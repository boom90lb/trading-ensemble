"""Shared score-to-book construction helpers.

The functions in this module deliberately stop at close-time target weights.
Execution timing, rebalance suppression against filled weights, borrow, and
capacity costs remain in ``src.execution.target_weights``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_book(targets: pd.DataFrame, max_gross: float, max_symbol_abs_weight: float) -> pd.DataFrame:
    """Scale rows over ``max_gross`` down proportionally, then clip per symbol.

    This never scales a low-gross row up. A directional scorer that emits only
    30% gross remains 30% gross, leaving the rest in cash.
    """
    if max_gross <= 0:
        raise ValueError(f"max_gross must be > 0, got {max_gross}")
    if max_symbol_abs_weight <= 0:
        raise ValueError(f"max_symbol_abs_weight must be > 0, got {max_symbol_abs_weight}")
    out = targets.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = out.abs().sum(axis=1)
    scale = (max_gross / gross.where(gross > max_gross)).fillna(1.0).clip(upper=1.0)
    return out.mul(scale, axis=0).clip(lower=-max_symbol_abs_weight, upper=max_symbol_abs_weight)


def apply_no_trade_band(targets: pd.DataFrame, band: float | pd.Series) -> pd.DataFrame:
    """Hold each name's weight until the target moves by more than ``band``."""
    if isinstance(band, pd.Series):
        thr = np.array([max(float(band.get(c, 0.0)), 0.0) for c in targets.columns], dtype=float)
    else:
        if band <= 0:
            return targets
        thr = np.full(targets.shape[1], float(band))
    mat = targets.to_numpy(dtype=float)
    out = np.empty_like(mat)
    held = np.zeros(mat.shape[1])
    for t in range(mat.shape[0]):
        tgt = mat[t]
        held = np.where(np.abs(tgt - held) > thr, tgt, held)
        out[t] = held
    return pd.DataFrame(out, index=targets.index, columns=targets.columns)


def cost_aware_band(
    half_life_bars: np.ndarray, per_trade_cost_frac: float, gamma: float = 1.0
) -> np.ndarray:
    """Heuristic no-trade half-width from OU speed and round-trip cost."""
    hl = np.asarray(half_life_bars, dtype=float)
    valid = np.isfinite(hl) & (hl > 0.0)
    kappa = np.where(valid, np.log(2.0) / np.where(valid, hl, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        band = gamma * np.sqrt(max(per_trade_cost_frac, 0.0) / kappa)
    return np.where(np.isfinite(band), band, 0.0)


def strength_multiplier(sscores: np.ndarray, entry_band: float, cap: float = 2.0) -> np.ndarray:
    """Conviction multiplier from absolute s-score distance past entry."""
    if entry_band <= 0:
        raise ValueError(f"entry_band must be > 0, got {entry_band}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    mult = (np.abs(np.asarray(sscores, dtype=float)) - entry_band) / entry_band
    return np.where(np.isfinite(mult), np.clip(mult, 0.0, cap), 0.0)


def build_residual_book_row(
    states: np.ndarray,
    beta_day: np.ndarray,
    eigenportfolios: np.ndarray,
    position_unit: float,
    size_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Net one residual-stat-arb bar into per-symbol target weights."""
    scale = 1.0 if size_scale is None else np.asarray(size_scale, dtype=float)
    stock_legs = position_unit * scale * np.asarray(states, dtype=float)
    factor_exposure = np.asarray(beta_day, dtype=float) @ stock_legs
    hedge_legs = -factor_exposure @ np.asarray(eigenportfolios, dtype=float)
    return stock_legs + hedge_legs


def construct_directional_targets(
    scores: pd.DataFrame,
    *,
    position_size: float,
    max_gross: float,
    max_symbol_abs_weight: float,
    no_trade_band: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Convert signed per-symbol convictions to unhedged target weights.

    ``scores`` are dimensionless convictions, typically in ``[-1, 1]``. The
    mapping is linear: ``target = position_size * score`` followed only by
    down-only gross/symbol caps. Missing scores stay missing so callers can use
    all-NaN rows as explicit fold-boundary no-op markers.
    """
    if position_size <= 0:
        raise ValueError(f"position_size must be > 0, got {position_size}")
    decision_mask = scores.notna()
    raw = (
        scores.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-1.0, upper=1.0)
        .fillna(0.0)
        * position_size
    )
    targets = cap_book(raw, max_gross=max_gross, max_symbol_abs_weight=max_symbol_abs_weight)
    if isinstance(no_trade_band, pd.Series) or no_trade_band > 0:
        targets = apply_no_trade_band(targets, no_trade_band)
        targets = cap_book(targets, max_gross=max_gross, max_symbol_abs_weight=max_symbol_abs_weight)
    return targets.where(decision_mask)
