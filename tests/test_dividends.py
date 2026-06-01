"""Dividend cash-credit tests (Phase 4 §4.1).

Prices are split- but NOT dividend-adjusted, so a held position takes a
mark-to-market markdown across an ex-date; the dividend credit (long) / debit
(short) offsets it, making the position total-return correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ExecutionConfig, TradingConfig
from src.models.base import BaseModel
from src.trading import Position, TradingStrategy


def _zero_cfg() -> ExecutionConfig:
    return ExecutionConfig(
        spread_bps=0.0,
        slippage_coeff=0.0,
        commission_bps=0.0,
        borrow_rate_bps_annual=0.0,
    )


class _ConstPositionModel(BaseModel):
    """Always emits the same target position so a trade opens deterministically."""

    def __init__(self, target: float):
        super().__init__(name="const")
        self.is_fitted = True
        self.target = target

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return np.full(len(X), self.target, dtype=float)

    def evaluate(self, X, y):
        return {}

    def save(self, path):
        return path

    def load(self, path):
        return self


def _strategy(target: float, initial: float = 10_000.0) -> TradingStrategy:
    return TradingStrategy(
        model=_ConstPositionModel(target),
        config=TradingConfig(
            initial_capital=initial, position_size=0.5, execution=_zero_cfg()
        ),
        sentiment_analyzer=None,
        use_sentiment=False,
    )


# --------------------------------------------------------------------------- #
# apply_dividends — direct
# --------------------------------------------------------------------------- #
def test_apply_dividends_long_credits_cash():
    s = _strategy(1.0)
    s._reset_state()
    s.positions["AAA"] = {"position": Position.LONG, "entry_price": 50.0, "size": 100.0}
    cash0 = s.cash
    ex = pd.Timestamp("2026-01-03")
    applied = s.apply_dividends({"AAA": pd.Series([0.5], index=[ex])}, ex)
    assert applied == pytest.approx(50.0)  # 100 shares * $0.50
    assert s.cash == pytest.approx(cash0 + 50.0)
    log = pd.DataFrame(s.transaction_log)
    assert (log["action"] == "DIVIDEND").sum() == 1
    assert log.iloc[-1]["net_pnl"] == pytest.approx(50.0)


def test_apply_dividends_short_debits_cash():
    s = _strategy(-1.0)
    s._reset_state()
    s.positions["AAA"] = {"position": Position.SHORT, "entry_price": 50.0, "size": 100.0}
    cash0 = s.cash
    ex = pd.Timestamp("2026-01-03")
    applied = s.apply_dividends({"AAA": pd.Series([0.5], index=[ex])}, ex)
    assert applied == pytest.approx(-50.0)  # short owes the dividend
    assert s.cash == pytest.approx(cash0 - 50.0)


def test_apply_dividends_no_position_is_noop():
    s = _strategy(1.0)
    s._reset_state()
    ex = pd.Timestamp("2026-01-03")
    assert s.apply_dividends({"AAA": pd.Series([0.5], index=[ex])}, ex) == 0.0
    assert s.cash == pytest.approx(10_000.0)


def test_apply_dividends_none_and_off_date_noop():
    s = _strategy(1.0)
    s._reset_state()
    s.positions["AAA"] = {"position": Position.LONG, "entry_price": 50.0, "size": 100.0}
    ex = pd.Timestamp("2026-01-03")
    series = {"AAA": pd.Series([0.5], index=[ex])}
    # None dividends -> no-op.
    assert s.apply_dividends(None, ex) == 0.0
    # A date with no dividend -> no-op.
    assert s.apply_dividends(series, pd.Timestamp("2026-01-04")) == 0.0
    assert s.cash == pytest.approx(10_000.0)


# --------------------------------------------------------------------------- #
# run_segment integration: dividend appears in the equity curve
# --------------------------------------------------------------------------- #
def _price_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    close = np.array([100.0, 100.0, 100.0, 99.0, 99.0, 99.0])  # ex-div markdown at idx[3]
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close},
        index=idx,
    )


def test_run_segment_credits_dividend_total_return_neutral():
    df = _price_frame()
    ex = df.index[3]
    divs = {"AAA": pd.Series([1.0], index=[ex], name="amount")}

    # With dividends: the $1 ex-date markdown is offset by the $1/share credit.
    s_div = _strategy(1.0)
    s_div.backtest(data={"AAA": df}, use_sentiment=False, dividends=divs)
    log = pd.DataFrame(s_div.transaction_log)
    div_rows = log[log["action"] == "DIVIDEND"]
    assert len(div_rows) == 1
    assert div_rows.iloc[0]["date"] == ex
    shares = float(div_rows.iloc[0]["shares"])
    assert shares > 0
    assert div_rows.iloc[0]["net_pnl"] == pytest.approx(shares * 1.0)

    # Same run without dividends: final equity is lower by exactly the credit
    # (positions/MTM are identical; only the dividend cash differs).
    s_nodiv = _strategy(1.0)
    s_nodiv.backtest(data={"AAA": df}, use_sentiment=False)
    pv_div = s_div.portfolio_value
    pv_nodiv = s_nodiv.portfolio_value
    assert pv_div - pv_nodiv == pytest.approx(shares * 1.0, rel=1e-6)
