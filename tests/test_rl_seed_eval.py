"""Phase 3.2 / M10 — RL overfitting controls.

Three concerns, all without training a real RL agent (the heavy fit is
exercised by the driver E2E offline, not in CI):

  * Reward audit (M10 pt3): TradingEnvironment.step reward uses only
    close[t+1] - close[t] (no future bars) and the documented formula.
  * Window feed (the run_segment fix this phase landed): the trading loop
    feeds windowed members their required_history rows and trades on the
    LAST predicted position — proven with a JAX-free stub.
  * Multi-seed aggregation (M10 pt1): summarize_member's mean +/- std / DSR
    over a fixed seed->returns mapping; held-out eval contract.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import EnsembleConfig, ExecutionConfig, ModelConfig, TradingConfig
from src.execution import ExecutionModel
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.models.lstm_ppo import LSTMPPO, TradingEnvironment
from src.trading import TradingStrategy

from scripts.rl_seed_eval import summarize_member


# --------------------------------------------------------------------------
# Reward audit (M10 part 3)
# --------------------------------------------------------------------------


def _reward_env(closes, window_size=2, transaction_cost=0.001):
    """TradingEnvironment over a hand-built close series (other OHLC = close)."""
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1_000_000] * len(closes),
    })
    return TradingEnvironment(
        df=df, features=["close"], window_size=window_size,
        transaction_cost=transaction_cost, reward_scaling=1.0,
    )


def test_reward_is_next_bar_return_net_of_turnover_cost():
    # window_size=2 → reset starts at step 2. closes chosen so each step's
    # next-bar return is exact.
    closes = [100.0, 100.0, 100.0, 110.0, 121.0]  # steps 2,3 score 0.10 each
    env = _reward_env(closes, window_size=2, transaction_cost=0.001)
    env.reset()
    assert env.current_step == 2

    # First action: position 1.0 from a starting position of 0.0.
    _, reward, _, _, _ = env.step(np.array([1.0]))
    # bar_return = (110 - 100)/100 = 0.10; turnover = |1 - 0| = 1.
    assert reward == pytest.approx(1.0 * 0.10 - 1.0 * 0.001)

    # Second action: hold position 1.0 → turnover 0 → reward is pure return.
    _, reward2, terminated, _, _ = env.step(np.array([1.0]))
    # bar_return = (121 - 110)/110 = 0.10; turnover = 0.
    assert reward2 == pytest.approx(1.0 * 0.10 - 0.0)
    assert terminated  # step 3 == len(df)-1 is terminal


def test_reward_uses_only_t_and_t_plus_1_no_future_peek():
    # Make the FUTURE (t+2 onward) wildly different; reward must not move.
    base = [100.0, 100.0, 100.0, 101.0]
    env_a = _reward_env(base + [1e6], window_size=2)        # huge future bar
    env_b = _reward_env(base + [0.01], window_size=2)       # tiny future bar
    env_a.reset()
    env_b.reset()
    _, r_a, _, _, _ = env_a.step(np.array([1.0]))
    _, r_b, _, _, _ = env_b.step(np.array([1.0]))
    # Both read close[2]=100 and close[3]=101 only → identical reward.
    assert r_a == pytest.approx(r_b)
    assert r_a == pytest.approx(1.0 * (101.0 - 100.0) / 100.0 - 0.001)


def test_reward_zero_position_is_only_turnover_cost():
    closes = [100.0, 100.0, 100.0, 150.0, 150.0]
    env = _reward_env(closes, window_size=2, transaction_cost=0.002)
    env.reset()
    # Action 0.0 from start 0.0 → turnover 0 → reward exactly 0 regardless of
    # the +50% next-bar move (we hold no position).
    _, reward, _, _, _ = env.step(np.array([0.0]))
    assert reward == pytest.approx(0.0)


def test_step_terminates_before_reading_past_the_end():
    # len=4, window_size=2 → only steps 2 is non-terminal; step from 2 lands
    # on 3 == len-1 → terminated, and a further step is a no-op (no t+1 read).
    env = _reward_env([100.0, 100.0, 100.0, 105.0], window_size=2)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0.5]))
    assert terminated
    # Stepping a done env returns 0 reward, no IndexError on a missing t+1.
    _, r, term2, _, _ = env.step(np.array([0.5]))
    assert r == 0.0 and term2 is True


# --------------------------------------------------------------------------
# required_history + window feed (the run_segment fix)
# --------------------------------------------------------------------------


def test_required_history_overrides():
    lstm_ppo = LSTMPPO(target_column="close", horizon=5)
    # RL agents need window_size + 2 (the env's (window, t, t+1) requirement).
    assert lstm_ppo.required_history == lstm_ppo.window_size + 2

    # A point-only ensemble (xgboost) needs exactly 1 row.
    point = EnsembleModel(
        target_column="close", horizon=5,
        config=EnsembleConfig(
            models=[ModelConfig(name="xgboost", weight=1.0)],
            weighting_strategy="static",
        ),
    )
    assert point.required_history == 1

    # An ensemble's requirement is the max over its members.
    mixed = EnsembleModel(
        target_column="close", horizon=5,
        config=EnsembleConfig(
            models=[
                ModelConfig(name="xgboost", weight=1.0),
                ModelConfig(name="lstm_ppo", weight=1.0),
            ],
            weighting_strategy="static",
        ),
    )
    assert mixed.required_history == lstm_ppo.window_size + 2


class _WindowSpyModel(BaseModel):
    """Needs 5 rows; returns a per-row vector whose LAST element is a strong
    LONG. Records how many rows it was fed each predict() call."""

    def __init__(self):
        super().__init__(name="spy", target_column="close", horizon=1)
        self.is_fitted = True
        self.fed_lengths: list[int] = []

    @property
    def required_history(self) -> int:
        return 5

    def fit(self, X, y=None, **k):
        return self

    def predict(self, X):
        self.fed_lengths.append(len(X))
        out = np.zeros(len(X))
        if len(out):
            out[-1] = 0.9  # last row → LONG
        return out

    def save(self, directory):
        return directory

    def load(self, path):
        return self


def _ramp_df(n=40):
    idx = pd.bdate_range("2022-01-03", periods=n, tz="America/New_York")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_run_segment_feeds_required_history_window_and_trades_on_last_row():
    df = _ramp_df(40)
    idx = df.index
    model = _WindowSpyModel()
    cfg = TradingConfig(
        initial_capital=10_000.0, position_size=0.5, signal_threshold=0.1,
        execution=ExecutionConfig(),
    )
    strat = TradingStrategy(
        model=model, config=cfg, use_sentiment=False,
        execution_model=ExecutionModel(ExecutionConfig()),
    )
    strat._reset_state()
    strat.run_segment(
        data={"AAPL": df}, start_date=str(idx[10]), end_date=str(idx[-1]),
        use_sentiment=False,
    )

    # Every predict() call was fed exactly required_history rows (not 1).
    assert set(model.fed_lengths) == {5}
    # The last-row LONG (0.9 > threshold 0.1) actually produced trades.
    assert len(strat.transaction_log) > 0


def test_run_segment_point_model_unchanged_single_row_feed():
    # required_history default 1 → the loop still feeds a single row, so point
    # models are bit-for-bit unchanged by the window fix.
    class _PointSpy(_WindowSpyModel):
        @property
        def required_history(self) -> int:
            return 1

    df = _ramp_df(20)
    model = _PointSpy()
    cfg = TradingConfig(initial_capital=10_000.0, signal_threshold=0.1,
                        execution=ExecutionConfig())
    strat = TradingStrategy(model=model, config=cfg, use_sentiment=False,
                            execution_model=ExecutionModel(ExecutionConfig()))
    strat._reset_state()
    strat.run_segment(data={"AAPL": df}, start_date=str(df.index[5]),
                      end_date=str(df.index[-1]), use_sentiment=False)
    assert set(model.fed_lengths) == {1}


# --------------------------------------------------------------------------
# Multi-seed aggregation (M10 part 1) — summarize_member
# --------------------------------------------------------------------------


def _gaussian_returns(seed, mu=0.001, sigma=0.01, n=300):
    return pd.Series(np.random.default_rng(seed).normal(mu, sigma, n))


def test_summarize_member_mean_std_over_seeds():
    seed_returns = {s: _gaussian_returns(s) for s in range(5)}
    out = summarize_member("lstm_ppo", seed_returns)
    assert out["n_seeds"] == 5 and out["n_valid"] == 5
    # mean/std are finite and consistent with the per-seed Sharpes.
    sharpes = np.array(list(out["seed_sharpe"].values()))
    assert out["mean"] == pytest.approx(float(sharpes.mean()))
    assert out["std"] == pytest.approx(float(sharpes.std(ddof=1)))
    assert out["dsr"] is not None  # DSR over the 5-seed trial set


def test_summarize_member_single_seed_has_no_dispersion():
    out = summarize_member("xlstm_ppo", {0: _gaussian_returns(0)})
    assert out["n_valid"] == 1
    assert out["mean"] is not None
    # One seed → no std to estimate, nothing to deflate.
    assert out["std"] is None
    assert out["dsr"] is None  # expected_max_sharpe is NaN for N<2


def test_summarize_member_empty_is_all_none():
    out = summarize_member("xlstm_grpo", {})
    assert out["n_valid"] == 0
    assert out["mean"] is None and out["std"] is None and out["dsr"] is None


def test_summarize_member_skips_degenerate_seed():
    # A flat (zero-variance) return series → NaN Sharpe → dropped from valid.
    seed_returns = {
        0: _gaussian_returns(0),
        1: pd.Series(np.zeros(300)),  # flat → NaN sharpe
        2: _gaussian_returns(2),
    }
    out = summarize_member("lstm_ppo", seed_returns)
    assert out["n_seeds"] == 3
    assert out["n_valid"] == 2  # the flat seed is excluded from mean/std/DSR
