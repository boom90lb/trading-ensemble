"""Map forecast outputs to position sizes via inverse-volatility scaling.

A forecast model emits ŷ_{t+h} (predicted price); the trading layer needs a
position ∈ [-1, 1]. The standard quant-finance bridge is to size the bet by
the forecast's expected return divided by recent realized volatility, then
clip. This makes ensemble members commensurable with policy-model positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, window: int = 20, vol_floor: float = 5e-3) -> np.ndarray:
    """Rolling std of log returns. Floored to avoid divide-by-zero on flat windows."""
    log_ret = np.log(close).diff()
    sigma = log_ret.rolling(window=window, min_periods=max(2, window // 4)).std()
    sigma = sigma.bfill().fillna(vol_floor).to_numpy()
    return np.maximum(sigma, vol_floor)


def forecast_to_position(
    forecast_price: np.ndarray,
    current_price: np.ndarray,
    sigma: np.ndarray,
    target_vol: float = 1.0,
    cap: float = 1.0,
) -> np.ndarray:
    """Convert forecast prices to positions sized by inverse realized vol.

    position_t = clip(target_vol * expected_return_t / sigma_t, -cap, cap)

    target_vol is in the same units as sigma (per-bar). With target_vol=1 the
    position equals the forecast Sharpe (clipped) — full-bet at one-sigma
    edge, half-bet at half-sigma, etc.
    """
    forecast_price = np.asarray(forecast_price, dtype=np.float64)
    current_price = np.asarray(current_price, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if forecast_price.shape != current_price.shape:
        raise ValueError(
            f"shape mismatch: forecast {forecast_price.shape} vs current {current_price.shape}"
        )
    if sigma.shape != current_price.shape:
        raise ValueError(f"shape mismatch: sigma {sigma.shape} vs current {current_price.shape}")

    expected_return = (forecast_price - current_price) / current_price
    raw_position = target_vol * expected_return / sigma
    return np.clip(raw_position, -cap, cap)


def ideal_position(
    realized_return: np.ndarray,
    sigma: np.ndarray,
    target_vol: float = 1.0,
    cap: float = 1.0,
) -> np.ndarray:
    """The bet a perfect-foresight trader would have placed.

    Used as the meta-learner regression target so the constrained weighted
    blend of model positions targets risk-adjusted realized return.
    """
    realized_return = np.asarray(realized_return, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    raw = target_vol * realized_return / sigma
    return np.clip(raw, -cap, cap)
