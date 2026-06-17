"""Phase 2.2: regression tests for the WFO outer-loop primitives.

Covers the surface area that the WFO loops in scripts/training.py and
scripts/backtest.py depend on:

  * ``TradingStrategy._reset_state`` zeroes per-backtest mutables.
  * ``TradingStrategy.run_segment`` does NOT reset state — state carries
    across consecutive calls, bar_idx continues, history accumulates.
  * ``TradingStrategy.backtest`` (the public one-shot) is exactly the
    composition ``_reset_state → run_segment → drop_pending → _finalize_results``.
  * ``_finalize_results`` computes daily_return / cumulative_return /
    drawdown over a concatenated multi-segment frame.
  * ``src.tracking.mlflow_utils.init_mlflow`` returns an experiment id and
    is idempotent; ``log_metrics_safe`` filters NaN/Inf without raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.baselines import BuyAndHold
from src.config import EnsembleConfig, ExecutionConfig, ModelConfig, TradingConfig
from src.execution import ExecutionModel
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.trading import TradingStrategy


def _make_gbm_df(n: int, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    """Synthetic OHLCV with positive close drift so BuyAndHold accumulates pnl."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.001, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    # tz-aware ET index: data_loader always localizes bars, and
    # clean_data_for_training asserts a tz-aware index (Phase 4 §4.2).
    idx = pd.bdate_range(start=start, periods=n, tz="America/New_York")
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )
    return df


def _write_fold_replay_artifacts(
    fold_dir: Path,
    *,
    symbol: str,
    horizon: int,
    target_col: str,
    usable: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    from scripts.training import build_fold_metadata
    from src.features import FeatureEngineer

    train_raw = usable.iloc[train_idx]
    test_raw = usable.iloc[test_idx]
    fe = FeatureEngineer()
    fe.fit_scalers(train_raw, symbol)
    train_scaled = fe.transform_features(train_raw, symbol, is_train=True)
    feature_columns = [
        c for c in train_scaled.columns
        if "target_" not in c and "direction_" not in c
    ]
    fe.save_state(
        fold_dir / "feature_engineer_state.pkl",
        symbol=symbol,
        feature_columns=feature_columns,
    )
    with open(fold_dir / "fold_metadata.json", "w") as f:
        json.dump(
            build_fold_metadata(
                symbol=symbol,
                horizon=horizon,
                target_col=target_col,
                train_raw=train_raw,
                test_raw=test_raw,
                model_feature_columns=feature_columns,
            ),
            f,
            indent=2,
        )


def _build_strategy() -> Tuple[TradingStrategy, pd.DataFrame, ExecutionConfig]:
    df = _make_gbm_df(n=60)
    bah = BuyAndHold()
    bah.fit(df, df["close"])
    bah.prepare(df["close"])
    exec_cfg = ExecutionConfig(
        spread_bps=1.0, slippage_coeff=0.0, commission_bps=1.0,
        borrow_rate_bps_annual=0.0, default_order_type="MOO",
    )
    cfg = TradingConfig(
        initial_capital=10_000.0, position_size=0.5,
        stop_loss=0.02, take_profit=0.05, execution=exec_cfg,
    )
    strat = TradingStrategy(
        model=bah, config=cfg, sentiment_analyzer=None,
        use_sentiment=False, execution_model=ExecutionModel(exec_cfg),
    )
    return strat, df, exec_cfg


def test_reset_state_zeros_all_mutables():
    strat, df, _ = _build_strategy()
    # Run one bar to populate state, then reset.
    strat.run_segment({"SYM": df.iloc[:10]}, use_sentiment=False)
    assert strat.history, "run_segment should append to history"

    strat._reset_state()

    assert strat.positions == {}
    assert strat.cash == strat.config.initial_capital
    assert strat.portfolio_value == strat.config.initial_capital
    assert strat.history == []
    assert strat.transaction_log == []
    assert strat._bar_dates == {}
    assert strat.execution_model.pending_count() == 0


def test_run_segment_does_not_reset_between_calls():
    """The whole point of the refactor: two run_segment() calls should
    behave like one continuous backtest, not two independent ones."""
    strat, df, _ = _build_strategy()
    strat._reset_state()

    seg1 = strat.run_segment({"SYM": df.iloc[:30]}, use_sentiment=False)
    cash_after_seg1 = strat.cash
    history_len_after_seg1 = len(strat.history)
    positions_after_seg1 = dict(strat.positions)

    # Second segment immediately after.
    seg2 = strat.run_segment({"SYM": df.iloc[30:]}, use_sentiment=False)

    # State persisted: cash/positions did NOT reset to initial_capital.
    # (BuyAndHold opens a LONG on bar 1; after seg1 we should be holding
    # an open LONG and cash should be reduced.)
    assert cash_after_seg1 < strat.config.initial_capital, (
        f"After seg1 cash should be reduced from initial; got {cash_after_seg1}"
    )
    assert positions_after_seg1, "Should be holding a LONG after seg1"
    # seg2 inherits seg1's open position — no re-OPEN_LONG transaction in seg2
    # unless we flatten in between. Assert that seg2 history is appended on
    # TOP of seg1 history.
    assert len(strat.history) == history_len_after_seg1 + len(seg2)
    assert not seg1.empty and not seg2.empty
    # Segments should not overlap in time.
    assert seg1.index.max() < seg2.index.min()


def test_run_segment_bar_idx_continues_across_calls():
    strat, df, _ = _build_strategy()
    strat._reset_state()

    strat.run_segment({"SYM": df.iloc[:20]}, use_sentiment=False)
    bars_after_seg1 = max(strat._bar_dates.keys())
    strat.run_segment({"SYM": df.iloc[20:40]}, use_sentiment=False)
    bars_after_seg2 = max(strat._bar_dates.keys())

    # The second segment's bar indices start at bars_after_seg1+1 (contiguous).
    assert bars_after_seg2 > bars_after_seg1
    # Sanity: bar_idx count == total bars processed.
    assert len(strat._bar_dates) == bars_after_seg2 + 1


def test_backtest_equals_manual_composition():
    """backtest() must be observably identical to manual reset + segment
    + drop_pending + finalize. This is the core regression guarantee."""
    strat_a, df, _ = _build_strategy()
    result_a = strat_a.backtest({"SYM": df}, use_sentiment=False)

    strat_b, _, _ = _build_strategy()
    strat_b._reset_state()
    seg = strat_b.run_segment({"SYM": df}, use_sentiment=False)
    strat_b.execution_model.drop_pending()
    result_b = TradingStrategy._finalize_results(seg)

    pd.testing.assert_frame_equal(result_a, result_b)
    # And the cash/position state at the end should match too.
    assert strat_a.cash == pytest.approx(strat_b.cash)
    assert strat_a.portfolio_value == pytest.approx(strat_b.portfolio_value)
    assert len(strat_a.transaction_log) == len(strat_b.transaction_log)


def test_finalize_results_derived_columns():
    seg = pd.DataFrame(
        {"portfolio_value": [100.0, 110.0, 99.0, 121.0], "cash": [50.0, 40.0, 30.0, 60.0]},
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    out = TradingStrategy._finalize_results(seg)
    assert list(out.columns) == ["portfolio_value", "cash", "daily_return",
                                  "cumulative_return", "drawdown"]
    # daily_return[1] = 110/100 - 1 = 0.10
    assert out["daily_return"].iloc[1] == pytest.approx(0.10)
    # cumulative_return[3] should reflect 121/100 - 1 = 0.21
    assert out["cumulative_return"].iloc[3] == pytest.approx(0.21)
    # drawdown is positive: 1 - portfolio/cummax. At idx=2: 1 - 99/110 = 0.10.
    assert out["drawdown"].iloc[2] == pytest.approx(1 - 99.0 / 110.0)
    # Idempotency: input frame is not mutated.
    assert "daily_return" not in seg.columns


def test_finalize_results_concatenated_segments_one_continuous_curve():
    """The cumulative_return + drawdown should treat concatenated segments
    as one continuous portfolio — this is the WFO loop's reporting story."""
    seg1 = pd.DataFrame(
        {"portfolio_value": [100.0, 105.0, 110.0], "cash": [50.0, 50.0, 50.0]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    seg2 = pd.DataFrame(
        {"portfolio_value": [110.0, 115.5, 99.0], "cash": [50.0, 50.0, 30.0]},
        index=pd.bdate_range("2024-01-08", periods=3),
    )
    concatenated = pd.concat([seg1, seg2])
    out = TradingStrategy._finalize_results(concatenated)

    # Final cumulative_return reflects 99/100 - 1 = -0.01 (from very first bar).
    assert out["cumulative_return"].iloc[-1] == pytest.approx(-0.01)
    # Max drawdown straddles the fold boundary: peak at 115.5, trough at 99.
    assert out["drawdown"].iloc[-1] == pytest.approx(1 - 99.0 / 115.5)


# -- Phase 2.5: conformal / ACI integration through the backtest loop ---------


def _make_conformal_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """GBM OHLCV + lag features, matching the conformal-fit fixture in
    test_conformal.py so the ensemble's dynamic-weight meta-learner populates
    a fitted EnbPI calibrator + ACI state."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B", tz="America/New_York")
    returns = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1000, 10000, n),
            "lag1_close": np.roll(close, 1),
            "lag2_close": np.roll(close, 2),
        },
        index=dates,
    )
    df.iloc[:2, df.columns.get_indexer(["lag1_close", "lag2_close"])] = close[0]
    return df


def _build_conformal_ensemble(df: pd.DataFrame, horizon: int = 5) -> EnsembleModel:
    y = df["close"].shift(-horizon).dropna()
    X = df.loc[y.index]
    ensemble = EnsembleModel(
        target_column="close",
        horizon=horizon,
        config=EnsembleConfig(
            models=[ModelConfig(name="xgboost", enabled=True, weight=1.0)],
            weighting_strategy="dynamic",
        ),
    )
    ensemble.meta_cv_folds = 3
    ensemble.fit(X, y)
    return ensemble


def test_run_segment_drives_aci_online_update(monkeypatch):
    """Phase 2.5 integration: an ensemble-with-conformal run through the
    backtest loop must FIRE ACI online updates. run_segment's per-symbol FIFO
    realizes each band h bars later against the realized ideal position (the
    same target the calibrator regressed on) and feeds the in-band outcome to
    ACIState.update — so alpha_t drifts off its target and realized coverage
    is observable. Without this loop, ACI is inert and the EnbPI+ACI choice
    collapses to fixed-alpha EnbPI."""
    df = _make_conformal_frame(n=300)
    ensemble = _build_conformal_ensemble(df, horizon=5)
    assert ensemble.has_conformal(), "fixture must produce a fitted calibrator"

    exec_cfg = ExecutionConfig(
        spread_bps=1.0, slippage_coeff=0.0, commission_bps=1.0,
        borrow_rate_bps_annual=0.0, default_order_type="MOO",
    )
    cfg = TradingConfig(
        initial_capital=10_000.0, position_size=0.5,
        stop_loss=0.02, take_profit=0.05, execution=exec_cfg,
    )
    strat = TradingStrategy(
        model=ensemble, config=cfg, sentiment_analyzer=None,
        use_sentiment=False, execution_model=ExecutionModel(exec_cfg),
    )

    # Spy on the ACI feedback to capture realized in-band outcomes without
    # duplicating the realization logic under test.
    captured: List[bool] = []
    orig_update = ensemble.update_aci

    def _spy(in_band: bool):
        captured.append(bool(in_band))
        return orig_update(in_band)

    monkeypatch.setattr(ensemble, "update_aci", _spy)

    a0 = ensemble.aci.current_alpha()
    result = strat.backtest({"SYM": df}, use_sentiment=False)

    assert not result.empty
    # n=300, horizon=5 → the FIFO must realize the overwhelming majority of bars.
    assert len(captured) > 250, "FIFO under-realized — ACI loop likely mis-wired"
    # Every ACI step moves alpha_t by a nonzero gamma-scaled amount, so a single
    # realized update is enough to leave the initial target.
    a1 = ensemble.aci.current_alpha()
    assert a1 != a0
    assert 0.0 < a1 < 1.0
    # Loose, non-flaky coverage sanity: the calibrated 90% bands should cover a
    # clear majority of realized ideal positions. A sign/scale regression in the
    # realization path would crater this well below 0.5.
    coverage = sum(captured) / len(captured)
    assert coverage >= 0.5


def test_run_segment_no_aci_updates_without_conformal():
    """The FIFO realization path must be a strict no-op for a non-conformal
    model (e.g. a baseline) — no AttributeError, no spurious state."""
    strat, df, _ = _build_strategy()  # BuyAndHold model, no conformal
    result = strat.backtest({"SYM": df}, use_sentiment=False)
    assert not result.empty


class _RecordingForecast(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="xgboost")
        self.is_fitted = True
        self.seen_close: List[List[float]] = []

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        self.seen_close.append([float(v) for v in X["close"].to_numpy()])
        return X["close"].to_numpy(dtype=float)

    def save(self, directory: Path) -> Path:
        return Path(directory)

    def load(self, model_path: Path):
        return self

    def evaluate(self, X_test, y_test):
        return {}


class _RecordingPolicy(BaseModel):
    def __init__(self) -> None:
        super().__init__(name="lstm_ppo")
        self.is_fitted = True
        self.seen_close: List[List[float]] = []

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        self.seen_close.append([float(v) for v in X["close"].to_numpy()])
        return np.full(len(X), 0.5, dtype=float)

    def save(self, directory: Path) -> Path:
        return Path(directory)

    def load(self, model_path: Path):
        return self

    def evaluate(self, X_test, y_test):
        return {}


def test_run_segment_keeps_policy_members_on_raw_model_frame():
    raw = _make_gbm_df(n=8, seed=41)
    scaled = raw.copy()
    scaled["open"] = np.linspace(0.10, 0.17, len(scaled))
    scaled["high"] = scaled["open"] + 0.01
    scaled["low"] = scaled["open"] - 0.01
    scaled["close"] = np.linspace(0.20, 0.27, len(scaled))

    forecast = _RecordingForecast()
    policy = _RecordingPolicy()
    ensemble = EnsembleModel(target_column="close", horizon=5, config=EnsembleConfig(models=[]))
    ensemble.models = {"xgboost": forecast, "lstm_ppo": policy}
    ensemble.weights = {"xgboost": 0.0, "lstm_ppo": 1.0}
    ensemble.kinds = {"xgboost": "forecast", "lstm_ppo": "policy"}
    ensemble.is_fitted = True

    exec_cfg = ExecutionConfig(
        spread_bps=0.0, slippage_coeff=0.0, commission_bps=0.0,
        borrow_rate_bps_annual=0.0, default_order_type="MOO",
    )
    cfg = TradingConfig(
        initial_capital=10_000.0, position_size=0.1,
        stop_loss=0.02, take_profit=0.05, execution=exec_cfg,
    )
    strat = TradingStrategy(
        model=ensemble, config=cfg, sentiment_analyzer=None,
        use_sentiment=False, execution_model=ExecutionModel(exec_cfg),
    )

    result = strat.run_segment(
        data={"SYM": raw},
        model_data={"SYM": scaled},
        use_sentiment=False,
    )

    assert not result.empty
    assert forecast.seen_close
    assert policy.seen_close
    assert max(v for seen in forecast.seen_close for v in seen) < 1.0
    assert min(v for seen in policy.seen_close for v in seen) > 1.0


# -- MLflow utility tests -----------------------------------------------------


def test_init_mlflow_returns_experiment_id(monkeypatch, tmp_path):
    # Redirect both the module-level constant AND the live mlflow URI.
    import src.tracking.mlflow_utils as mu
    monkeypatch.setattr(mu, "MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")

    exp_id = mu.init_mlflow("wfo_pipeline_test")
    assert isinstance(exp_id, str) and exp_id

    # Idempotent: same experiment, same id.
    exp_id2 = mu.init_mlflow("wfo_pipeline_test")
    assert exp_id == exp_id2


def test_log_metrics_safe_filters_non_finite(monkeypatch, tmp_path):
    import src.tracking.mlflow_utils as mu
    monkeypatch.setattr(mu, "MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mu.init_mlflow("wfo_log_filter_test")

    with mlflow.start_run() as run:
        mu.log_metrics_safe(
            {
                "good": 1.5,
                "nan": float("nan"),
                "inf": float("inf"),
                "neg_inf": float("-inf"),
                "non_numeric": "abc",
                "none": None,
            },
            step=0,
            prefix="fold0_",
        )

    client = mlflow.tracking.MlflowClient()
    logged = client.get_run(run.info.run_id).data.metrics
    assert "fold0_good" in logged
    assert logged["fold0_good"] == pytest.approx(1.5)
    # The non-finite / non-numeric ones must have been silently filtered.
    for bad in ("fold0_nan", "fold0_inf", "fold0_neg_inf",
                "fold0_non_numeric", "fold0_none"):
        assert bad not in logged


def test_train_symbol_wfo_produces_per_fold_artifacts(monkeypatch, tmp_path):
    """End-to-end smoke for scripts/training.py's per-symbol WFO loop.

    Uses a 200-bar synthetic frame and an xgboost-only static-weighted
    ensemble (no policy members + no meta-learner OOF) so the fit budget
    stays under a few seconds. Verifies that:

      * Each fold writes ``ensemble_model/`` + ``split_idx.npz``.
      * The saved split indices are non-overlapping across folds.
      * ``fold_metrics.json`` aggregates per-fold + mean/std metrics.
      * Per-fold metrics are logged to MLflow under the nested run with
        a ``step=fold_idx`` index so the run timeline is queryable.
    """
    import types

    import src.tracking.mlflow_utils as mu
    monkeypatch.setattr(mu, "MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    exp_id = mu.init_mlflow("wfo_training_smoke")

    from scripts.training import build_features, train_symbol_wfo
    from src.config import EnsembleConfig, ModelConfig, TrainingConfig
    from src.features import FeatureEngineer
    from src.validation.walk_forward import PurgedWalkForward

    raw = _make_gbm_df(n=200, seed=7)
    fe = FeatureEngineer()
    full_df = build_features(
        raw_df=raw, symbol="SYM",
        feature_engineer=fe, sentiment_analyzer=None,
        horizon=5,
    )
    assert "target_5" in full_df.columns

    training_config = TrainingConfig(
        symbols=["SYM"], timeframe="1d",
        start_date="2024-01-02", prediction_horizon=5,
        n_splits=2, embargo_pct=0.01, expanding=True,
    )
    splitter = PurgedWalkForward(
        n_splits=training_config.n_splits,
        purge_horizon=training_config.effective_purge_horizon,
        embargo_pct=training_config.embargo_pct,
        expanding=training_config.expanding,
    )
    model_configs = [ModelConfig(name="xgboost", enabled=True, weight=1.0)]

    # Override ensemble weighting to "static" so meta-learner OOF is skipped
    # (keeps the test under ~5s by avoiding meta_cv_folds * xgboost refits).
    import scripts.training as training_mod
    orig_ensemble = training_mod.EnsembleModel

    class _StaticEnsemble(orig_ensemble):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # Force static weighting regardless of caller intent.
            self.config = EnsembleConfig(
                models=self.config.models, weighting_strategy="static",
            )
    monkeypatch.setattr(training_mod, "EnsembleModel", _StaticEnsemble)

    symbol_dir = tmp_path / "out" / "SYM"
    symbol_dir.mkdir(parents=True)

    args = types.SimpleNamespace(
        rl_timesteps=0, checkpoint_freq=0, gpu=False,
    )

    with mlflow.start_run(experiment_id=exp_id, run_name="parent"):
        with mlflow.start_run(run_name="SYM", nested=True) as nested:
            per_fold = train_symbol_wfo(
                symbol="SYM",
                full_df=full_df,
                target_col="target_5",
                splitter=splitter,
                model_configs=model_configs,
                model_names=["xgboost"],
                training_config=training_config,
                args=args,
                gpu_available=False,
                feature_engineer=fe,
                symbol_dir=symbol_dir,
            )
            nested_run_id = nested.info.run_id

    # Each fold's artifacts exist and split indices are non-overlapping.
    fold_dirs = sorted(d for d in symbol_dir.glob("fold_*") if d.is_dir())
    assert len(fold_dirs) >= 2, f"expected >=2 folds, got {len(fold_dirs)}"

    test_idx_sets = []
    for d in fold_dirs:
        assert (d / "ensemble_model").is_dir()
        assert (d / "feature_engineer_state.pkl").exists()
        assert (d / "fold_metadata.json").exists()
        sidx = np.load(d / "split_idx.npz")
        assert sidx["train_idx"].size > 0 and sidx["test_idx"].size > 0
        test_idx_sets.append(set(sidx["test_idx"].tolist()))
    for i in range(len(test_idx_sets)):
        for j in range(i + 1, len(test_idx_sets)):
            assert not (test_idx_sets[i] & test_idx_sets[j]), (
                f"fold {i} and {j} test indices overlap"
            )

    # fold_metrics.json present + aggregated.
    metrics_path = symbol_dir / "fold_metrics.json"
    assert metrics_path.exists()
    import json
    with open(metrics_path) as f:
        payload = json.load(f)
    assert len(payload["per_fold"]) == len(fold_dirs) == len(per_fold)
    assert any(k.startswith("mean_") for k in payload["aggregated"]), (
        f"expected mean_<metric> keys in aggregated; got {payload['aggregated'].keys()}"
    )

    # MLflow received per-fold metrics on the nested run.
    client = mlflow.tracking.MlflowClient()
    nested_run = client.get_run(nested_run_id)
    fold_metric_names = [k for k in nested_run.data.metrics
                         if k.startswith("fold/")]
    assert fold_metric_names, "expected at least one fold/* metric on nested run"
    # History on a step-indexed metric should have one point per fold.
    sample_metric = fold_metric_names[0]
    history = client.get_metric_history(nested_run_id, sample_metric)
    assert {h.step for h in history} >= set(range(len(fold_dirs)))


def test_backtest_wfo_persists_state_and_drops_pending_between_folds(
    monkeypatch, tmp_path,
):
    """The backtest-side WFO loop must:

      * Reset state ONCE before fold 0 (not between folds).
      * Call ``execution_model.drop_pending()`` exactly once per fold.
      * Swap ``strategy.model`` via ``load_fold_model`` at each fold k>0.
      * Concatenate per-fold segments into one continuous equity curve.

    Uses a BuyAndHold baseline through ``run_symbol_wfo`` with two fake
    fold dirs whose ``split_idx.npz`` files carve up the 200-bar synthetic
    frame into halves. No ensemble pickles needed.
    """
    import types

    from scripts.backtest import _fold_date_range, run_symbol_wfo
    from scripts.training import build_features
    from src.execution import ExecutionModel
    from src.features import FeatureEngineer

    raw = _make_gbm_df(n=200, seed=13)
    fe = FeatureEngineer()
    full_df = build_features(
        raw_df=raw, symbol="SYM",
        feature_engineer=fe, sentiment_analyzer=None,
        horizon=5,
    )
    usable = full_df.dropna(subset=["target_5"]).copy()

    # Two fake folds: first half + second half of `usable`.
    n = len(usable)
    fold0_test = np.arange(0, n // 2)
    fold1_test = np.arange(n // 2, n)
    train_stub = np.arange(0, 10)

    fold_root = tmp_path / "training_run" / "SYM"
    fold_dirs = []
    for i, test_idx in enumerate([fold0_test, fold1_test]):
        d = fold_root / f"fold_{i}"
        (d / "ensemble_model").mkdir(parents=True, exist_ok=True)
        np.savez(d / "split_idx.npz",
                 train_idx=train_stub, test_idx=test_idx)
        fold_dirs.append(d)

    # Confirm the date-range helper resolves the slices we expect.
    r0 = _fold_date_range(fold_dirs[0], usable)
    r1 = _fold_date_range(fold_dirs[1], usable)
    assert r0 is not None and r1 is not None
    assert r0[1] < r1[0]  # non-overlapping in time

    # Count drop_pending() invocations on the execution model the strategy
    # ends up using. We patch the class method so the count includes the
    # invocation that happens inside the strategy's TradingStrategy.__init__
    # path as well as fold-boundary calls.
    drop_calls = {"n": 0}
    orig_drop = ExecutionModel.drop_pending

    def _counting_drop(self):
        drop_calls["n"] += 1
        return orig_drop(self)

    monkeypatch.setattr(ExecutionModel, "drop_pending", _counting_drop)

    baseline = BuyAndHold()
    baseline.fit(full_df, full_df["close"])
    baseline.prepare(full_df["close"])

    args = types.SimpleNamespace(
        initial_capital=10_000.0, position_size=0.5,
        stop_loss=0.02, take_profit=0.05,
        commission_bps=1.0, spread_bps=1.0, slippage_coeff=0.0,
        borrow_bps_annual=0.0, order_type="MOO",
    )

    drop_baseline = drop_calls["n"]
    result = run_symbol_wfo(
        symbol="SYM",
        model=baseline,
        fold_dirs=fold_dirs,
        full_df=full_df,
        usable=usable,
        args=args,
        use_sentiment=False,
        sentiment_analyzer=None,
        load_fold_model=None,
        horizon=None,
    )
    fold_boundary_drops = drop_calls["n"] - drop_baseline

    # 1 drop from _reset_state() at strategy init + 2 drops at fold
    # boundaries = 3 total. Allow >=2 fold-boundary drops to keep the
    # assertion loose against future state-reset call sites.
    assert fold_boundary_drops >= 3, (
        f"expected >=3 drop_pending calls (1 reset + 2 fold boundaries); "
        f"got {fold_boundary_drops}"
    )

    # Equity curve covers BOTH folds — concatenation worked.
    assert not result["equity_curve"].empty
    assert len(result["fold_metrics"]) == 2
    assert all(fm["n_bars"] > 0 for fm in result["fold_metrics"])
    # Total bars >= bars in fold 0 alone (sanity: concat ran).
    n_bars_total = sum(fm["n_bars"] for fm in result["fold_metrics"])
    assert len(result["equity_curve"]) >= n_bars_total - 2


def test_backtest_wfo_swaps_model_per_fold(monkeypatch, tmp_path):
    """When ``load_fold_model`` is supplied, the strategy's ``.model`` is
    swapped at every fold k>=1. We assert this by tracking calls."""
    import types

    from scripts.backtest import run_symbol_wfo
    from scripts.training import build_features
    from src.features import FeatureEngineer

    raw = _make_gbm_df(n=120, seed=21)
    fe = FeatureEngineer()
    full_df = build_features(
        raw_df=raw, symbol="SYM",
        feature_engineer=fe, sentiment_analyzer=None, horizon=5,
    )
    usable = full_df.dropna(subset=["target_5"]).copy()
    n = len(usable)

    fold_root = tmp_path / "training_run" / "SYM"
    fold_dirs = []
    for i, test_slice in enumerate(
        [np.arange(0, n // 3),
         np.arange(n // 3, 2 * n // 3),
         np.arange(2 * n // 3, n)]
    ):
        d = fold_root / f"fold_{i}"
        (d / "ensemble_model").mkdir(parents=True, exist_ok=True)
        train_idx = np.arange(0, 5)
        np.savez(
            d / "split_idx.npz",
            train_idx=train_idx, test_idx=test_slice,
        )
        _write_fold_replay_artifacts(
            d,
            symbol="SYM",
            horizon=5,
            target_col="target_5",
            usable=usable,
            train_idx=train_idx,
            test_idx=test_slice,
        )
        fold_dirs.append(d)

    # Make 3 distinct baseline instances; load_fold_model returns one
    # baseline per fold_dir lookup so we can verify swaps.
    baselines = [BuyAndHold() for _ in range(3)]
    for b in baselines:
        b.fit(full_df, full_df["close"])
        b.prepare(full_df["close"])

    load_calls: List[int] = []

    def _fake_loader(fold_dir: Path, horizon: int) -> BuyAndHold:
        idx = int(fold_dir.name.split("_")[-1])
        load_calls.append(idx)
        return baselines[idx]

    args = types.SimpleNamespace(
        initial_capital=10_000.0, position_size=0.5,
        stop_loss=0.02, take_profit=0.05,
        commission_bps=1.0, spread_bps=1.0, slippage_coeff=0.0,
        borrow_bps_annual=0.0, order_type="MOO",
    )

    result = run_symbol_wfo(
        symbol="SYM",
        model=baselines[0],
        fold_dirs=fold_dirs,
        full_df=full_df,
        usable=usable,
        args=args,
        use_sentiment=False,
        sentiment_analyzer=None,
        load_fold_model=_fake_loader,
        horizon=5,
    )

    # load_fold_model is called only for fold_idx >= 1 (fold 0 uses the
    # initial `model` passed in). So 2 calls for 3 folds.
    assert load_calls == [1, 2], f"unexpected load order: {load_calls}"
    assert len(result["fold_metrics"]) == 3
    assert not result["equity_curve"].empty


def test_fold_metadata_validates_replay_index(tmp_path):
    from scripts.backtest import _fold_date_range
    from scripts.training import build_features
    from src.features import FeatureEngineer

    raw = _make_gbm_df(n=90, seed=31)
    fe = FeatureEngineer()
    full_df = build_features(
        raw_df=raw, symbol="SYM",
        feature_engineer=fe, sentiment_analyzer=None, horizon=5,
    )
    usable = full_df.dropna(subset=["target_5"]).copy()
    train_idx = np.concatenate([np.arange(0, 20), np.arange(25, 30)])
    test_idx = np.arange(40, 60)
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    np.savez(fold_dir / "split_idx.npz", train_idx=train_idx, test_idx=test_idx)
    _write_fold_replay_artifacts(
        fold_dir,
        symbol="SYM",
        horizon=5,
        target_col="target_5",
        usable=usable,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    resolved = _fold_date_range(
        fold_dir,
        usable,
        require_metadata=True,
        symbol="SYM",
        horizon=5,
        target_col="target_5",
    )
    assert resolved == (usable.index[test_idx[0]], usable.index[test_idx[-1]])

    drifted = usable.drop(index=usable.index[test_idx[3]])
    with pytest.raises(ValueError, match="Test index missing|Test index hash mismatch"):
        _fold_date_range(
            fold_dir,
            drifted,
            require_metadata=True,
            symbol="SYM",
            horizon=5,
            target_col="target_5",
        )


def test_missing_fold_metadata_fails_loud_for_ensemble_replay(tmp_path):
    from scripts.backtest import _fold_date_range
    from scripts.training import build_features
    from src.features import FeatureEngineer

    raw = _make_gbm_df(n=60, seed=32)
    fe = FeatureEngineer()
    full_df = build_features(
        raw_df=raw, symbol="SYM",
        feature_engineer=fe, sentiment_analyzer=None, horizon=5,
    )
    usable = full_df.dropna(subset=["target_5"]).copy()
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    np.savez(
        fold_dir / "split_idx.npz",
        train_idx=np.arange(0, 20),
        test_idx=np.arange(20, 30),
    )

    with pytest.raises(FileNotFoundError, match="R0 fold replay metadata"):
        _fold_date_range(
            fold_dir,
            usable,
            require_metadata=True,
            symbol="SYM",
            horizon=5,
            target_col="target_5",
        )


def test_log_params_safe_coerces_to_strings(monkeypatch, tmp_path):
    import src.tracking.mlflow_utils as mu
    monkeypatch.setattr(mu, "MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mu.init_mlflow("wfo_log_params_test")

    with mlflow.start_run() as run:
        mu.log_params_safe(
            {"int_p": 5, "float_p": 3.14, "list_p": [1, 2, 3],
             "none_p": None, "str_p": "x"}
        )

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run.info.run_id).data.params
    assert params["int_p"] == "5"
    assert params["float_p"] == "3.14"
    assert params["list_p"] == "[1, 2, 3]"
    assert params["str_p"] == "x"
    # None is skipped, not stored as "None".
    assert "none_p" not in params
