"""Property tests for ExecutionModel + Trader fill semantics + costs.

Verifies the rigor-load-bearing pieces:
  * No same-bar fill: signal at bar t fills at bar t+1.
  * Slippage is adverse on both sides (buys above ref, sells below).
  * Commission and borrow formulas match the documented bps math.
  * Fold integrity: an unfilled order at the last bar of fold A must NOT
    spill into fold B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ExecutionConfig, TradingConfig
from src.execution import ExecutionModel, Order, OrderType
from src.execution.costs import (
    commission_dollars,
    daily_borrow_dollars,
    slippage_bps,
)
from src.models.base import BaseModel
from src.trading import TradingStrategy


# ---------- pure cost functions ----------


def test_slippage_zero_size_is_half_spread() -> None:
    assert slippage_bps(0.0, 100_000.0, half_spread_bps=1.0, slippage_coeff=10.0) == 1.0


def test_slippage_linear_in_notional() -> None:
    # 50%-of-portfolio trade with coeff=10 → impact 5 bps + spread 1 bp = 6 bps.
    got = slippage_bps(50_000.0, 100_000.0, half_spread_bps=1.0, slippage_coeff=10.0)
    assert got == pytest.approx(6.0)


def test_slippage_uses_abs_notional() -> None:
    # Sells (negative notional) cost the same as buys of the same size.
    buy = slippage_bps(50_000.0, 100_000.0, 1.0, 10.0)
    sell = slippage_bps(-50_000.0, 100_000.0, 1.0, 10.0)
    assert buy == sell


def test_commission_bps_arithmetic() -> None:
    # 10 bps of $10,000 = $10.
    assert commission_dollars(10_000.0, 10.0) == pytest.approx(10.0)
    # Sign-insensitive.
    assert commission_dollars(-10_000.0, 10.0) == pytest.approx(10.0)


def test_daily_borrow_no_short_zero() -> None:
    assert daily_borrow_dollars(0.0, 50.0) == 0.0


def test_daily_borrow_formula() -> None:
    # 50 bps annual on $10,000 short, 252 days: 10000 * 0.005 / 252 ≈ 0.1984
    got = daily_borrow_dollars(10_000.0, 50.0, trading_days_per_year=252)
    assert got == pytest.approx(10_000.0 * 0.005 / 252)


# ---------- ExecutionModel: fill timing ----------


def _bar(open_: float, close_: float) -> dict:
    return {"AAA": {"open": open_, "close": close_}}


def _cfg(**kw) -> ExecutionConfig:
    base = dict(
        spread_bps=0.0, slippage_coeff=0.0, commission_bps=0.0,
        borrow_rate_bps_annual=0.0,
    )
    base.update(kw)
    return ExecutionConfig(**base)


def test_moo_fills_at_next_bar_open() -> None:
    em = ExecutionModel(_cfg())
    em.submit(Order("AAA", side=1, shares=10.0, order_type=OrderType.MOO, submit_bar=3))
    # Same-bar step (bar 3) must NOT fill.
    fills = em.step(bar_idx=3, bar_ohlc=_bar(100.0, 101.0), portfolio_value=100_000.0)
    assert fills == []
    assert em.pending_count() == 1
    # Next bar fills at the open.
    fills = em.step(bar_idx=4, bar_ohlc=_bar(102.0, 103.0), portfolio_value=100_000.0)
    assert len(fills) == 1
    assert fills[0].fill_price == 102.0
    assert em.pending_count() == 0


def test_moc_fills_at_next_bar_close() -> None:
    em = ExecutionModel(_cfg())
    em.submit(Order("AAA", side=1, shares=10.0, order_type=OrderType.MOC, submit_bar=0))
    fills = em.step(bar_idx=1, bar_ohlc=_bar(50.0, 55.0), portfolio_value=100_000.0)
    assert len(fills) == 1
    assert fills[0].fill_price == 55.0


def test_buy_slippage_adverse() -> None:
    em = ExecutionModel(_cfg(spread_bps=10.0))
    em.submit(Order("AAA", side=1, shares=10.0, order_type=OrderType.MOO, submit_bar=0))
    fill = em.step(1, _bar(100.0, 101.0), portfolio_value=100_000.0)[0]
    # 10 bps half-spread → fill 0.10% above ref.
    assert fill.fill_price == pytest.approx(100.0 * 1.001)
    assert fill.fill_price > 100.0


def test_sell_slippage_adverse() -> None:
    em = ExecutionModel(_cfg(spread_bps=10.0))
    em.submit(Order("AAA", side=-1, shares=10.0, order_type=OrderType.MOO, submit_bar=0))
    fill = em.step(1, _bar(100.0, 99.0), portfolio_value=100_000.0)[0]
    assert fill.fill_price == pytest.approx(100.0 * 0.999)
    assert fill.fill_price < 100.0


def test_linear_impact_scales_with_size() -> None:
    em = ExecutionModel(_cfg(spread_bps=0.0, slippage_coeff=10.0))
    # 100-share buy at $100 ref = $10k notional. Portfolio $10k → trade = 100% of pv.
    # Impact = 10 bps. Fill price = 100 * (1 + 0.001) = 100.10
    em.submit(Order("AAA", side=1, shares=100.0, order_type=OrderType.MOO, submit_bar=0))
    fill = em.step(1, _bar(100.0, 100.0), portfolio_value=10_000.0)[0]
    assert fill.fill_price == pytest.approx(100.10)


def test_missing_symbol_keeps_order_queued() -> None:
    em = ExecutionModel(_cfg())
    em.submit(Order("AAA", side=1, shares=1.0, order_type=OrderType.MOO, submit_bar=0))
    # Step with OHLC for a different symbol — should not fill, but should stay queued.
    fills = em.step(
        bar_idx=1,
        bar_ohlc={"BBB": {"open": 50.0, "close": 50.0}},
        portfolio_value=10_000.0,
    )
    assert fills == []
    assert em.pending_count() == 1


def test_drop_pending_clears_queue() -> None:
    em = ExecutionModel(_cfg())
    em.submit(Order("AAA", side=1, shares=1.0, order_type=OrderType.MOO, submit_bar=0))
    em.submit(Order("AAA", side=-1, shares=1.0, order_type=OrderType.MOO, submit_bar=0))
    dropped = em.drop_pending()
    assert dropped == 2
    assert em.pending_count() == 0


def test_fold_boundary_orders_do_not_leak() -> None:
    """Submit at last bar of fold A → drop_pending() → fold B sees nothing."""
    em = ExecutionModel(_cfg())
    em.submit(Order("AAA", side=1, shares=5.0, order_type=OrderType.MOO, submit_bar=99))
    # End of fold A: drop unfilled orders.
    em.drop_pending()
    # Fold B starts. The order from fold A must not magically fill here.
    fills = em.step(bar_idx=0, bar_ohlc=_bar(50.0, 51.0), portfolio_value=10_000.0)
    assert fills == []
    assert em.pending_count() == 0


# ---------- TradingStrategy integration: no same-bar fills ----------


class _ConstPositionModel(BaseModel):
    """Predicts a constant ensemble position regardless of input."""

    def __init__(self, target_position: float) -> None:
        super().__init__(name="const_pos")
        self.target_position = target_position
        self.is_fitted = True

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return np.full(len(X), self.target_position, dtype=float)

    def evaluate(self, X, y):
        return {}

    def save(self, path):
        return path

    def load(self, path):
        return self


@pytest.fixture
def simple_data() -> dict:
    # 5 bars of monotone-up OHLC for symbol AAA.
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=idx,
    )
    return {"AAA": df}


def _strategy(target_position: float, exec_cfg: ExecutionConfig) -> TradingStrategy:
    return TradingStrategy(
        model=_ConstPositionModel(target_position),
        config=TradingConfig(
            initial_capital=10_000.0,
            position_size=0.5,
            execution=exec_cfg,
        ),
        sentiment_analyzer=None,
        use_sentiment=False,
    )


def test_strategy_signal_fills_at_next_open_not_same_bar(simple_data) -> None:
    """Bar 0 generates LONG signal; bar 1 is where the OPEN_LONG transaction lands."""
    exec_cfg = _cfg()  # zero costs for clean accounting
    s = _strategy(target_position=1.0, exec_cfg=exec_cfg)
    s.backtest(data=simple_data, use_sentiment=False)

    log = pd.DataFrame(s.transaction_log)
    opens = log[log["action"] == "OPEN_LONG"]
    assert len(opens) >= 1
    first_open = opens.iloc[0]
    # First signal is generated at bar 0 (2026-01-01) close; fill is at bar 1 open.
    assert first_open["date"] == pd.Timestamp("2026-01-02")
    # Fill price is bar-1 open = 101.0 (no slippage in this config).
    assert first_open["price"] == pytest.approx(101.0)


def test_strategy_long_to_short_flip_emits_close_then_open(simple_data) -> None:
    """A target-position sign flip queues CLOSE_LONG then OPEN_SHORT on the same bar."""

    class _FlippingModel(BaseModel):
        def __init__(self):
            super().__init__(name="flip")
            self.is_fitted = True
            self.calls = 0

        def fit(self, X, y=None, **kwargs):
            return self

        def predict(self, X):
            self.calls += 1
            # First three signals LONG, then SHORT.
            val = 1.0 if self.calls <= 3 else -1.0
            return np.full(len(X), val, dtype=float)

        def evaluate(self, X, y): return {}
        def save(self, path): return path
        def load(self, path): return self

    s = TradingStrategy(
        model=_FlippingModel(),
        config=TradingConfig(
            initial_capital=10_000.0,
            position_size=0.5,
            execution=_cfg(),
        ),
        sentiment_analyzer=None,
        use_sentiment=False,
    )
    s.backtest(data=simple_data, use_sentiment=False)

    log = pd.DataFrame(s.transaction_log)
    actions = log["action"].tolist()
    # Open long after bar 0; then on the flip-signal bar two fills appear on the
    # following bar: CLOSE_LONG first, then OPEN_SHORT.
    assert "OPEN_LONG" in actions
    assert "CLOSE_LONG" in actions
    assert "OPEN_SHORT" in actions
    close_idx = actions.index("CLOSE_LONG")
    open_short_idx = actions.index("OPEN_SHORT")
    assert close_idx < open_short_idx


def test_strategy_borrow_charged_on_open_short(simple_data) -> None:
    exec_cfg = _cfg(borrow_rate_bps_annual=50.0)
    s = _strategy(target_position=-1.0, exec_cfg=exec_cfg)
    s.backtest(data=simple_data, use_sentiment=False)

    log = pd.DataFrame(s.transaction_log)
    borrow_rows = log[log["action"] == "BORROW"]
    # Short opens on bar 1 fill; bars 1..4 carry the short, so we expect at
    # least one borrow row after the short exists.
    assert len(borrow_rows) >= 1
    assert (borrow_rows["borrow_cost"] > 0).all()


def test_strategy_no_orders_left_pending_after_backtest(simple_data) -> None:
    s = _strategy(target_position=1.0, exec_cfg=_cfg())
    s.backtest(data=simple_data, use_sentiment=False)
    assert s.execution_model.pending_count() == 0
