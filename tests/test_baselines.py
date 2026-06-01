"""Tests for baseline strategies (BuyAndHold, MACrossover, TSMOM).

Covers:
  * Deterministic position output from rule semantics (BAH always +1,
    MA-X flips at known crossovers, TSMOM signs match K-bar return).
  * Warmup-period FLAT (lookback windows not yet filled).
  * Predict-before-prepare returns FLAT (no crash).
  * Integration through TradingStrategy.backtest — baselines flow through
    the same ExecutionModel + cost machinery as the ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import BuyAndHold, MACrossover, TSMOM
from src.config import ExecutionConfig, TradingConfig
from src.trading import TradingStrategy


# ---------- BuyAndHold ----------


def test_buy_and_hold_predict_always_one() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    X = pd.DataFrame({"close": np.arange(10.0)}, index=idx)
    bh = BuyAndHold()
    bh.prepare(X["close"])
    out = bh.predict(X)
    assert out.shape == (10,)
    assert np.all(out == 1.0)


def test_buy_and_hold_predict_without_prepare_still_works() -> None:
    # BAH has no state to prepare; predict is unconditional.
    bh = BuyAndHold()
    X = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2026-01-01", periods=1))
    assert bh.predict(X).tolist() == [1.0]


# ---------- MACrossover ----------


def test_ma_crossover_rejects_fast_geq_slow() -> None:
    with pytest.raises(ValueError):
        MACrossover(fast=50, slow=20)
    with pytest.raises(ValueError):
        MACrossover(fast=20, slow=20)


def test_ma_crossover_predict_before_prepare_returns_flat() -> None:
    ma = MACrossover(fast=5, slow=10)
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    X = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    out = ma.predict(X)
    assert np.all(out == 0.0)


def test_ma_crossover_warmup_is_flat() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    close = pd.Series(np.arange(100.0, 120.0), index=idx)  # rising
    ma = MACrossover(fast=5, slow=10)
    ma.prepare(close)
    # First slow-1=9 bars have no slow MA → FLAT.
    assert ma._positions is not None
    assert np.all(ma._positions.iloc[:9] == 0.0)


def test_ma_crossover_long_on_rising_series() -> None:
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    close = pd.Series(np.arange(100.0, 130.0), index=idx)
    ma = MACrossover(fast=5, slow=10)
    ma.prepare(close)
    # Once both MAs are defined, fast > slow on a strictly rising series.
    assert ma._positions is not None
    assert np.all(ma._positions.iloc[10:] == 1.0)


def test_ma_crossover_short_on_falling_series() -> None:
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    close = pd.Series(np.arange(130.0, 100.0, -1.0), index=idx)
    ma = MACrossover(fast=5, slow=10)
    ma.prepare(close)
    assert ma._positions is not None
    assert np.all(ma._positions.iloc[10:] == -1.0)


def test_ma_crossover_predict_is_date_lookup() -> None:
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    close = pd.Series(np.arange(100.0, 130.0), index=idx)
    ma = MACrossover(fast=5, slow=10)
    ma.prepare(close)
    # Predict on the full frame should match _positions exactly.
    X = pd.DataFrame({"close": close.values}, index=idx)
    np.testing.assert_array_equal(ma.predict(X), ma._positions.values)


# ---------- TSMOM ----------


def test_tsmom_rejects_zero_lookback() -> None:
    with pytest.raises(ValueError):
        TSMOM(lookback=0)


def test_tsmom_predict_before_prepare_returns_flat() -> None:
    ts = TSMOM(lookback=5)
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    X = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    assert np.all(ts.predict(X) == 0.0)


def test_tsmom_warmup_is_flat() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    close = pd.Series(np.arange(100.0, 120.0), index=idx)
    ts = TSMOM(lookback=10)
    ts.prepare(close)
    assert ts._positions is not None
    # First `lookback` bars have NaN pct_change → FLAT after fillna.
    assert np.all(ts._positions.iloc[:10] == 0.0)


def test_tsmom_long_on_rising_series() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    close = pd.Series(np.arange(100.0, 120.0), index=idx)
    ts = TSMOM(lookback=5)
    ts.prepare(close)
    assert ts._positions is not None
    # After warmup, 5-bar return is positive → LONG.
    assert np.all(ts._positions.iloc[5:] == 1.0)


def test_tsmom_short_on_falling_series() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    close = pd.Series(np.arange(120.0, 100.0, -1.0), index=idx)
    ts = TSMOM(lookback=5)
    ts.prepare(close)
    assert ts._positions is not None
    assert np.all(ts._positions.iloc[5:] == -1.0)


def test_tsmom_flat_on_constant_series() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    close = pd.Series(np.full(20, 100.0), index=idx)
    ts = TSMOM(lookback=5)
    ts.prepare(close)
    assert ts._positions is not None
    assert np.all(ts._positions == 0.0)


# ---------- Integration via TradingStrategy.backtest ----------


def _exec_cfg() -> ExecutionConfig:
    return ExecutionConfig(
        spread_bps=0.0, slippage_coeff=0.0, commission_bps=0.0,
        borrow_rate_bps_annual=0.0,
    )


def _rising_data(n: int = 30) -> dict:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = np.linspace(100.0, 130.0, n)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
        },
        index=idx,
    )
    return {"AAA": df}


def _make_strategy(model) -> TradingStrategy:
    return TradingStrategy(
        model=model,
        config=TradingConfig(
            initial_capital=10_000.0,
            position_size=0.5,
            execution=_exec_cfg(),
        ),
        sentiment_analyzer=None,
        use_sentiment=False,
    )


def test_buy_and_hold_integration_opens_long() -> None:
    data = _rising_data(n=10)
    bh = BuyAndHold()
    bh.fit(data["AAA"], data["AAA"]["close"])
    bh.prepare(data["AAA"]["close"])

    s = _make_strategy(bh)
    results = s.backtest(data=data, use_sentiment=False)
    assert not results.empty

    log = pd.DataFrame(s.transaction_log)
    # Should have at least one OPEN_LONG; never an OPEN_SHORT.
    actions = log["action"].tolist() if not log.empty else []
    assert "OPEN_LONG" in actions
    assert "OPEN_SHORT" not in actions


def test_ma_crossover_integration_runs_clean() -> None:
    data = _rising_data(n=40)
    ma = MACrossover(fast=3, slow=8)
    ma.fit(data["AAA"], data["AAA"]["close"])
    ma.prepare(data["AAA"]["close"])

    s = _make_strategy(ma)
    results = s.backtest(data=data, use_sentiment=False)
    assert not results.empty
    # Rising series → eventually LONG. No exception, no leftover pending orders.
    assert s.execution_model.pending_count() == 0


def test_tsmom_integration_runs_clean() -> None:
    data = _rising_data(n=20)
    ts = TSMOM(lookback=5)
    ts.fit(data["AAA"], data["AAA"]["close"])
    ts.prepare(data["AAA"]["close"])

    s = _make_strategy(ts)
    results = s.backtest(data=data, use_sentiment=False)
    assert not results.empty
    assert s.execution_model.pending_count() == 0
    # Rising series after warmup → LONG → must have entered a long.
    log = pd.DataFrame(s.transaction_log)
    actions = log["action"].tolist() if not log.empty else []
    assert "OPEN_LONG" in actions


def test_baselines_are_per_symbol_independent() -> None:
    """Each baseline instance binds to one symbol's close. Prepare on a
    different series must change the position lookup."""
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    rising = pd.Series(np.arange(100.0, 130.0), index=idx)
    falling = pd.Series(np.arange(130.0, 100.0, -1.0), index=idx)

    ma_up = MACrossover(fast=5, slow=10)
    ma_up.prepare(rising)
    ma_dn = MACrossover(fast=5, slow=10)
    ma_dn.prepare(falling)

    # After warmup, opposite signals.
    assert ma_up._positions is not None and ma_dn._positions is not None
    assert ma_up._positions.iloc[-1] == 1.0
    assert ma_dn._positions.iloc[-1] == -1.0
