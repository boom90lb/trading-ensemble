"""Capacity-aware (ADV-scaled) price impact on the target-weight engine (plan 003, re-applied).

main's portfolio backtester (src/execution/target_weights.py) charges impact keyed on a
name's portfolio-weight fraction, not its liquidity, so the broad book charges ~0 impact
regardless of how illiquid a name is. This adds an opt-in term keyed on participation
(traded $ / trailing ADV $). Default ``adv_impact_coeff=0`` reproduces current numbers;
the tests cover parity-off, the illiquidity penalty, and the floor that bounds a
near-zero-ADV name. (Re-implemented against main's per-row ``_cost_row`` engine after the
residual-statarb merge; the branch's original patched the old vectorized backtester.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.execution.target_weights import backtest_target_weights
from src.config import ExecutionConfig

_FREE = dict(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0)


def _single_name_impact(
    adv_value: float,
    coeff: float,
    floor: float = 1.0e5,
    capital: float = 1.0e7,
    model: str = "linear",
) -> float:
    """Total impact cost for one name entered long, isolating the ADV term."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    open_prices = pd.DataFrame({"X": [100.0, 101.0, 102.0, 103.0]}, index=idx)
    weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0]}, index=idx)  # one entry trade (fills under main's pending logic)
    dollar_volume = pd.DataFrame({"X": [adv_value] * 4}, index=idx)
    execution = ExecutionConfig(
        adv_impact_coeff=coeff,
        adv_impact_model=model,
        adv_floor_dollars=floor,
        **_FREE,
    )
    result = backtest_target_weights(
        open_prices, weights, execution=execution, initial_capital=capital, dollar_volume=dollar_volume
    )
    return float(result.costs["impact"].sum())


def test_adv_impact_off_is_parity() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    open_prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [50.0, 49.0, 51.0, 52.0, 53.0]}, index=idx)
    weights = pd.DataFrame({"A": [0.3] * 5, "B": [-0.2] * 5}, index=idx)
    dollar_volume = pd.DataFrame({"A": [1.0e8] * 5, "B": [1.0e8] * 5}, index=idx)

    base = backtest_target_weights(open_prices, weights)  # default config, no ADV panel
    with_panel = backtest_target_weights(open_prices, weights, dollar_volume=dollar_volume)  # coeff 0 + panel

    # adv_impact_coeff defaults to 0, so supplying the panel must change nothing.
    pd.testing.assert_series_equal(base.returns, with_panel.returns)
    pd.testing.assert_frame_equal(base.costs, with_panel.costs)


def test_adv_impact_penalizes_illiquid_more() -> None:
    liquid = _single_name_impact(adv_value=1.0e8, coeff=5.0)
    illiquid = _single_name_impact(adv_value=1.0e6, coeff=5.0)  # 100x less ADV
    assert illiquid > liquid
    assert illiquid == pytest.approx(100.0 * liquid, rel=1e-9)  # linear in participation


def test_adv_impact_model_default_is_linear() -> None:
    implicit = _single_name_impact(adv_value=1.0e7, coeff=5.0)
    explicit = _single_name_impact(adv_value=1.0e7, coeff=5.0, model="linear")
    assert implicit == pytest.approx(explicit, rel=1e-12)


def test_sqrt_adv_impact_is_sublinear_in_participation() -> None:
    low = _single_name_impact(adv_value=1.0e8, coeff=5.0, capital=1.0e6, model="sqrt")
    high = _single_name_impact(adv_value=1.0e8, coeff=5.0, capital=1.0e8, model="sqrt")
    low_linear = _single_name_impact(adv_value=1.0e8, coeff=5.0, capital=1.0e6, model="linear")
    high_linear = _single_name_impact(adv_value=1.0e8, coeff=5.0, capital=1.0e8, model="linear")

    assert high > low
    assert high / low == pytest.approx(10.0, rel=1e-9)
    assert high_linear / low_linear == pytest.approx(100.0, rel=1e-9)


def test_adv_floor_caps_blowup() -> None:
    near_zero = _single_name_impact(adv_value=1.0, coeff=5.0, floor=1.0e5)
    at_floor = _single_name_impact(adv_value=1.0e5, coeff=5.0, floor=1.0e5)
    assert np.isfinite(near_zero)
    assert near_zero == pytest.approx(at_floor, rel=1e-9)  # near-zero ADV clipped to the floor
