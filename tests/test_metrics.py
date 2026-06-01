"""Tests for src/validation/metrics.py.

Covers:
  * PSR formula on synthetic returns (high SR / normal returns ⇒ high PSR).
  * PSR degenerate inputs (n<3, constant returns) ⇒ NaN.
  * Calmar uses the positive-max_drawdown convention; non-positive ⇒ NaN.
  * PBO sanity: identical strategies ⇒ PBO ≈ 0.5; one dominant strategy
    ⇒ PBO ≪ 0.5; one anti-overfit strategy (best IS = worst OOS by design)
    ⇒ PBO ≈ 1.
  * PBO undefined for <2 strategies and for too-short series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.validation.metrics import (
    calmar_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
    probability_backtest_overfitting,
)


# ---------- periodic Sharpe ----------


def test_periodic_sharpe_known_case() -> None:
    # Constant 1% return with zero variance → undefined (NaN).
    assert math.isnan(periodic_sharpe(np.full(100, 0.01)))


def test_periodic_sharpe_normal_returns() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=2000)
    sr = periodic_sharpe(returns)
    # True periodic Sharpe = 0.1; sample SR should land close.
    assert sr == pytest.approx(0.1, abs=0.03)


def test_periodic_sharpe_too_few_obs() -> None:
    assert math.isnan(periodic_sharpe(np.array([0.01])))


# ---------- PSR ----------


def test_psr_high_sr_normal_returns_close_to_one() -> None:
    rng = np.random.default_rng(1)
    # True periodic SR ≈ 0.2, n=2000 → PSR(0) should be ~1.
    returns = rng.normal(loc=0.002, scale=0.01, size=2000)
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    assert psr > 0.99


def test_psr_zero_edge() -> None:
    # Mean-zero noise: PSR(0) should be ~0.5.
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.0, scale=0.01, size=5000)
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    assert 0.4 <= psr <= 0.6


def test_psr_against_higher_benchmark_smaller() -> None:
    rng = np.random.default_rng(3)
    # True periodic SR = 0.1. Test against a benchmark of 0.0 vs 0.2.
    returns = rng.normal(loc=0.001, scale=0.01, size=2000)
    psr_0 = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    psr_high = probabilistic_sharpe_ratio(returns, sr_benchmark=0.2)
    assert psr_0 > psr_high


def test_psr_degenerate_inputs_return_nan() -> None:
    assert math.isnan(probabilistic_sharpe_ratio(np.array([0.01])))
    assert math.isnan(probabilistic_sharpe_ratio(np.full(100, 0.01)))


# ---------- Expected-max Sharpe (False Strategy Theorem) ----------


def test_expected_max_sharpe_grows_with_n() -> None:
    # Same dispersion, more trials → higher expected max under the null.
    rng = np.random.default_rng(20)
    small = rng.normal(0.0, 0.05, size=5)
    large = rng.normal(0.0, 0.05, size=200)
    # Hold dispersion fixed by rescaling to identical std.
    small = small / small.std(ddof=1) * 0.05
    large = large / large.std(ddof=1) * 0.05
    assert expected_max_sharpe(large) > expected_max_sharpe(small)


def test_expected_max_sharpe_scales_with_dispersion() -> None:
    base = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert expected_max_sharpe(2 * base) == pytest.approx(
        2 * expected_max_sharpe(base), rel=1e-9
    )


def test_expected_max_sharpe_identical_trials_zero() -> None:
    # No dispersion → no selection advantage to deflate.
    assert expected_max_sharpe(np.full(16, 0.3)) == 0.0


def test_expected_max_sharpe_single_trial_nan() -> None:
    assert math.isnan(expected_max_sharpe(np.array([0.5])))


# ---------- DSR ----------


def test_dsr_below_psr_when_multiple_trials() -> None:
    """Deflation must penalize: with N>1 dispersed trials, DSR < PSR(0)."""
    rng = np.random.default_rng(21)
    returns = rng.normal(loc=0.0015, scale=0.01, size=1500)
    trial_sharpes = rng.normal(0.05, 0.04, size=18)  # the sweep's 18-trial grid
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(returns, trial_sharpes)
    assert dsr < psr


def test_dsr_equals_psr_when_trials_identical() -> None:
    """Zero dispersion ⇒ sr_null = 0 ⇒ DSR collapses to PSR(0)."""
    rng = np.random.default_rng(22)
    returns = rng.normal(loc=0.001, scale=0.01, size=1500)
    trial_sharpes = np.full(18, 0.1)
    dsr = deflated_sharpe_ratio(returns, trial_sharpes)
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    assert dsr == pytest.approx(psr, abs=1e-12)


def test_dsr_monotonic_decreasing_in_trial_count() -> None:
    """Same returns + dispersion, more trials ⇒ harsher deflation ⇒ lower DSR."""
    rng = np.random.default_rng(23)
    returns = rng.normal(loc=0.0015, scale=0.01, size=2000)
    few = rng.normal(0.05, 0.04, size=4)
    many = rng.normal(0.05, 0.04, size=64)
    few = few / few.std(ddof=1) * 0.04
    many = many / many.std(ddof=1) * 0.04
    assert deflated_sharpe_ratio(returns, many) < deflated_sharpe_ratio(returns, few)


def test_dsr_nan_on_single_trial() -> None:
    rng = np.random.default_rng(24)
    returns = rng.normal(0.001, 0.01, size=500)
    assert math.isnan(deflated_sharpe_ratio(returns, np.array([0.2])))


def test_dsr_nan_on_degenerate_returns() -> None:
    assert math.isnan(deflated_sharpe_ratio(np.full(100, 0.01), np.arange(10) / 10))


def test_dsr_strong_strategy_survives_tight_cluster() -> None:
    """A genuinely strong selected strategy against tightly-clustered weak
    trials still clears the deflation (high DSR); the same strategy against a
    widely-dispersed lucky-max trial set is deflated much harder."""
    rng = np.random.default_rng(25)
    returns = rng.normal(loc=0.003, scale=0.01, size=2000)  # strong edge
    tight = rng.normal(0.0, 0.01, size=18)
    wide = rng.normal(0.0, 0.20, size=18)
    dsr_tight = deflated_sharpe_ratio(returns, tight)
    dsr_wide = deflated_sharpe_ratio(returns, wide)
    assert dsr_tight > dsr_wide
    assert dsr_tight > 0.9


# ---------- Calmar ----------


def test_calmar_positive_dd_convention() -> None:
    # 10% annual return on 5% max drawdown → 2.0.
    assert calmar_ratio(0.10, 0.05) == pytest.approx(2.0)


def test_calmar_nan_on_nonpositive_dd() -> None:
    assert math.isnan(calmar_ratio(0.10, 0.0))
    assert math.isnan(calmar_ratio(0.10, -0.05))


def test_calmar_nan_on_nonfinite_inputs() -> None:
    assert math.isnan(calmar_ratio(float("nan"), 0.05))
    assert math.isnan(calmar_ratio(0.1, float("inf")))


# ---------- PBO via CSCV ----------


def test_pbo_identical_strategies_is_half() -> None:
    """If all N strategies have the same returns, IS-best is essentially a
    coin flip; expected OOS rank percentile = 0.5; PBO ≈ 0.5."""
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.01, size=500)
    M = np.column_stack([base + rng.normal(0, 1e-8, size=500) for _ in range(4)])
    pbo = probability_backtest_overfitting(M, n_splits=10)
    assert 0.35 <= pbo <= 0.65


def test_pbo_one_dominant_strategy_low() -> None:
    """A strategy that is uniformly better than the rest should have a low PBO."""
    rng = np.random.default_rng(5)
    n = 500
    # Strategy 0 has a real edge; 1-3 are pure noise.
    M = np.column_stack([
        rng.normal(0.002, 0.01, size=n),
        rng.normal(0.0, 0.01, size=n),
        rng.normal(0.0, 0.01, size=n),
        rng.normal(0.0, 0.01, size=n),
    ])
    pbo = probability_backtest_overfitting(M, n_splits=10)
    assert pbo < 0.3


def test_pbo_anti_overfit_high() -> None:
    """Construct returns where the best-on-IS is engineered to be worst-on-OOS:
    flip the sign of every odd-slice for strategy 0 so it 'wins' on even
    slices and 'loses' on odd ones. PBO should be elevated."""
    rng = np.random.default_rng(6)
    T = 500
    n_splits = 10
    slice_len = T // n_splits
    base = rng.normal(0.0, 0.01, size=T)
    # Strategy 0: positive returns in even slices, negative in odd slices.
    s0 = base.copy()
    for i in range(n_splits):
        sign = 1 if i % 2 == 0 else -1
        s0[i * slice_len:(i + 1) * slice_len] = sign * 0.005 + rng.normal(0, 0.001, slice_len)
    M = np.column_stack([s0, rng.normal(0.0, 0.01, T), rng.normal(0.0, 0.01, T), rng.normal(0.0, 0.01, T)])
    pbo = probability_backtest_overfitting(M, n_splits=n_splits)
    # When IS picks the half where s0 was positive, OOS sees the negative half
    # → s0 ranks worst OOS → overfit indicator fires on many splits.
    assert pbo > 0.5


def test_pbo_single_strategy_undefined() -> None:
    M = np.random.default_rng(7).normal(0, 0.01, size=(500, 1))
    assert math.isnan(probability_backtest_overfitting(M, n_splits=10))


def test_pbo_rejects_odd_splits() -> None:
    M = np.random.default_rng(8).normal(0, 0.01, size=(500, 3))
    with pytest.raises(ValueError):
        probability_backtest_overfitting(M, n_splits=7)


def test_pbo_returns_nan_when_series_too_short() -> None:
    M = np.random.default_rng(9).normal(0, 0.01, size=(15, 3))
    # 15 obs / 10 splits = slice_len 1 < 2 → NaN.
    assert math.isnan(probability_backtest_overfitting(M, n_splits=10))


def test_pbo_accepts_dataframe() -> None:
    rng = np.random.default_rng(10)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(500, 4)),
                      columns=["a", "b", "c", "d"])
    pbo = probability_backtest_overfitting(df, n_splits=10)
    assert 0.0 <= pbo <= 1.0
