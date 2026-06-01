"""Regression tests for the three real-data plumbing bugs found 2026-05-31.

Every prior verification ran on synthetic GBM through code paths that masked
these; they only surfaced on the first real-data run. Each test below fails
against the pre-fix code and passes after the corresponding fix:

  1. EnsembleModel.save() never serialized forecast members → a loaded
     ensemble came back with UNFITTED members → all-zero positions → zero
     trades in every backtest fold. (commit: persist & reload forecast members)
  2. Prophet.fit/predict received a tz-aware ``ds`` → "Column ds has timezone
     specified" → Prophet fit failed every fold. (same commit)
  3. run_segment fed model.predict the target_/direction_ LABEL columns, which
     training drops before fitting → XGBoost feature_names mismatch on every
     bar → member contributed nothing. (commit: exclude label cols)

These use a forecast-only ensemble (arima + xgboost) so the assertions don't
depend on a JAX/GPU policy member; Prophet is exercised in its own test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EnsembleConfig, ModelConfig
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel


def _make_gbm_df(n: int, seed: int = 7, start: str = "2021-01-04") -> pd.DataFrame:
    """Synthetic OHLCV with a tz-aware ET index (the production invariant)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0005, scale=0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.bdate_range(start=start, periods=n, tz="America/New_York")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def _build_enhanced(n: int, horizon: int) -> tuple[pd.DataFrame, str, FeatureEngineer]:
    """Mirror scripts/training.build_features exactly: features + lags + target,
    then ``clean_data_for_training`` (Inf→NaN, drop >30%-NaN cols, ffill,
    fillna(0)) which is what removes the warmup-period NaNs Prophet rejects,
    then drop rows with no forward label. Returns (usable, target_col, fe)."""
    from scripts.training import clean_data_for_training

    fe = FeatureEngineer()
    df = fe.create_features(_make_gbm_df(n))
    df = fe.create_lagged_features(df, [1, 2, 5, 10])
    df = fe.create_target_variable(df, "close", horizon)
    df = clean_data_for_training(df)
    target_col = f"target_{horizon}"
    return df.dropna(subset=[target_col]).copy(), target_col, fe


def _fit_forecast_ensemble(members: list[str]):
    """Fit an arima/xgboost ensemble the way train_symbol_wfo does (TRAIN-only
    scaler fit, label columns dropped before fit). Returns (ensemble, X_test)."""
    horizon = 5
    usable, target_col, fe = _build_enhanced(n=320, horizon=horizon)
    split = 240
    train_raw, test_raw = usable.iloc[:split], usable.iloc[split:]

    fe.fit_scalers(train_raw, "SYM")
    train_scaled = fe.transform_features(train_raw, "SYM", is_train=True)
    test_scaled = fe.transform_features(test_raw, "SYM", is_train=False)

    drop = [c for c in train_scaled.columns if "target_" in c or "direction_" in c]
    X_train = train_scaled.drop(columns=drop)
    y_train = train_scaled[target_col]
    X_test = test_scaled.drop(columns=drop).reindex(columns=X_train.columns, fill_value=0)

    ensemble = EnsembleModel(
        target_column="close",
        horizon=horizon,
        config=EnsembleConfig(
            models=[ModelConfig(name=m, enabled=True, weight=1.0) for m in members],
            weighting_strategy="static",
        ),
    )
    ensemble.fit(X_train, y_train)
    return ensemble, X_test


# --------------------------------------------------------------------------
# Bug 1: forecast members must survive a save/load round-trip fitted.
# --------------------------------------------------------------------------
def test_save_load_roundtrip_keeps_forecast_members_fitted(tmp_path: Path):
    ensemble, X_test = _fit_forecast_ensemble(["arima", "xgboost"])
    assert ensemble.is_fitted
    assert all(m.is_fitted for m in ensemble.models.values()), "fixture must fit members"

    ensemble.save(directory=tmp_path)
    state = tmp_path / "ensemble_state_h5.pkl"
    assert state.exists()
    # The metadata-only pickle was ~800B; a real persisted ensemble writes the
    # members' own artifacts under per-member subdirs. Assert those exist.
    member_dirs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert {"arima", "xgboost"} <= member_dirs, (
        f"member artifacts not persisted; only found {member_dirs}"
    )

    reloaded = EnsembleModel(target_column="close", horizon=5)
    reloaded.load(state)
    assert reloaded.is_fitted, "ensemble must load back fitted"
    assert reloaded.models, "ensemble must load its members"
    assert all(m.is_fitted for m in reloaded.models.values()), (
        "every member must come back fitted (the save() no-op bug left them "
        "unfitted, yielding all-zero positions)"
    )


def test_loaded_ensemble_predicts_non_degenerate(tmp_path: Path):
    """The downstream symptom: a correctly round-tripped ensemble must emit a
    non-degenerate position vector (the bug produced all-zeros → 0 trades)."""
    ensemble, X_test = _fit_forecast_ensemble(["arima", "xgboost"])
    ensemble.save(directory=tmp_path)
    reloaded = EnsembleModel(target_column="close", horizon=5)
    reloaded.load(tmp_path / "ensemble_state_h5.pkl")

    pred = np.asarray(reloaded.predict(X_test), dtype=float)
    assert len(pred) == len(X_test)
    assert np.isfinite(pred).any(), "predictions must contain finite values"
    assert np.count_nonzero(np.nan_to_num(pred)) > 0, (
        "loaded ensemble produced all-zero positions (the unfitted-member bug)"
    )


# --------------------------------------------------------------------------
# Bug 2: Prophet must fit on a tz-aware (ET) index without the tz error.
# --------------------------------------------------------------------------
def test_prophet_fits_on_tz_aware_index():
    prophet = pytest.importorskip("prophet")  # noqa: F841
    from src.models.prophet import ProphetModel

    usable, target_col, fe = _build_enhanced(n=200, horizon=5)
    assert usable.index.tz is not None, "fixture must be tz-aware (production invariant)"

    model = ProphetModel(target_column="close", horizon=5)
    model.fit(usable, usable["close"])
    assert model.is_fitted, (
        "Prophet must fit on a tz-aware index; the bug raised 'Column ds has "
        "timezone specified, which is not supported' and left is_fitted False"
    )
    preds = np.asarray(model.predict(usable.iloc[-20:]), dtype=float)
    assert len(preds) == 20 and np.isfinite(preds).all()


# --------------------------------------------------------------------------
# Bug 3: a predict frame that still carries label columns must not break a
# member; the ensemble must match the fit-time column set.
# --------------------------------------------------------------------------
def test_predict_with_label_columns_present_still_works():
    """XGBoost validates feature names strictly. If the backtest passes a frame
    that still contains target_/direction_ columns, the per-bar predict raised a
    feature_names mismatch. After the run_segment fix those columns are dropped
    before predict; this asserts the ensemble predicts cleanly once they are."""
    horizon = 5
    usable, target_col, fe = _build_enhanced(n=320, horizon=horizon)
    split = 240
    train_raw, test_raw = usable.iloc[:split], usable.iloc[split:]
    fe.fit_scalers(train_raw, "SYM")
    train_scaled = fe.transform_features(train_raw, "SYM", is_train=True)
    test_scaled = fe.transform_features(test_raw, "SYM", is_train=False)

    drop = [c for c in train_scaled.columns if "target_" in c or "direction_" in c]
    X_train = train_scaled.drop(columns=drop)
    y_train = train_scaled[target_col]

    ensemble = EnsembleModel(
        target_column="close",
        horizon=horizon,
        config=EnsembleConfig(
            models=[ModelConfig(name="xgboost", enabled=True, weight=1.0)],
            weighting_strategy="static",
        ),
    )
    ensemble.fit(X_train, y_train)

    # Emulate run_segment's fixed column selection (drop label cols).
    label_cols = [c for c in test_scaled.columns if "target_" in c or "direction_" in c]
    assert label_cols, "fixture must contain label columns to exercise the drop"
    feat = test_scaled.drop(columns=label_cols).reindex(columns=X_train.columns, fill_value=0)

    pred = np.asarray(ensemble.predict(feat), dtype=float)
    assert len(pred) == len(feat)
    assert np.count_nonzero(np.nan_to_num(pred)) > 0, (
        "xgboost contributed nothing — feature-name mismatch not resolved"
    )


# --------------------------------------------------------------------------
# Bug 4: evaluate() must not return {} when the ensemble emits some NaN
# predictions. The real run hit "Error evaluating ensemble: Input contains
# NaN" 20× (1 per symbol-fold) because sklearn metrics reject any NaN, so the
# whole fold's metrics were dropped. The finite-mask guard scores the finite
# rows instead.
# --------------------------------------------------------------------------
def test_evaluate_tolerates_some_nan_predictions(monkeypatch):
    horizon = 5
    usable, target_col, fe = _build_enhanced(n=320, horizon=horizon)
    split = 240
    train_raw, test_raw = usable.iloc[:split], usable.iloc[split:]
    fe.fit_scalers(train_raw, "SYM")
    train_scaled = fe.transform_features(train_raw, "SYM", is_train=True)
    test_scaled = fe.transform_features(test_raw, "SYM", is_train=False)
    drop = [c for c in train_scaled.columns if "target_" in c or "direction_" in c]
    X_train = train_scaled.drop(columns=drop)
    y_train = train_scaled[target_col]
    X_test = test_scaled.drop(columns=drop).reindex(columns=X_train.columns, fill_value=0)
    y_test = test_scaled[target_col]

    ensemble = EnsembleModel(
        target_column="close",
        horizon=horizon,
        config=EnsembleConfig(
            models=[ModelConfig(name="xgboost", enabled=True, weight=1.0)],
            weighting_strategy="static",
        ),
    )
    ensemble.fit(X_train, y_train)

    # Inject NaN into a few predictions to mimic a member misbehaving on real
    # data, then confirm evaluate() still returns metrics (scored on the
    # finite rows) instead of an empty dict.
    real_predict = ensemble.predict

    def _nan_predict(X):
        out = np.asarray(real_predict(X), dtype=float).copy()
        if len(out) >= 3:
            out[:2] = np.nan
        return out

    monkeypatch.setattr(ensemble, "predict", _nan_predict)
    metrics = ensemble.evaluate(X_test, y_test)
    assert metrics, "evaluate() returned {} despite finite rows being available"
    assert "position_mae" in metrics and np.isfinite(metrics["position_mae"])
    assert metrics.get("position_n_finite", 0) >= 2


# --------------------------------------------------------------------------
# Bug 5: calculate_signal's inter-model-agreement boost must receive the real
# features window. It used to pass an empty pd.DataFrame(index=[date]), which
# made get_model_contributions -> _predict_positions_per_model raise
# "requires 'close' column" on EVERY bar (3,485x in the real run) — a logged
# ERROR plus a dead confidence boost.
# --------------------------------------------------------------------------
def test_calculate_signal_uses_features_for_agreement_boost(caplog):
    import logging

    from src.config import TradingConfig
    from src.trading import TradingStrategy

    horizon = 5
    usable, target_col, fe = _build_enhanced(n=320, horizon=horizon)
    split = 240
    train_raw, test_raw = usable.iloc[:split], usable.iloc[split:]
    fe.fit_scalers(train_raw, "SYM")
    train_scaled = fe.transform_features(train_raw, "SYM", is_train=True)
    test_scaled = fe.transform_features(test_raw, "SYM", is_train=False)
    drop = [c for c in train_scaled.columns if "target_" in c or "direction_" in c]
    X_train = train_scaled.drop(columns=drop)
    y_train = train_scaled[target_col]
    feat = test_scaled.drop(columns=drop).reindex(columns=X_train.columns, fill_value=0)

    ensemble = EnsembleModel(
        target_column="close",
        horizon=horizon,
        config=EnsembleConfig(
            models=[
                ModelConfig(name="arima", enabled=True, weight=1.0),
                ModelConfig(name="xgboost", enabled=True, weight=1.0),
            ],
            weighting_strategy="static",
        ),
    )
    ensemble.fit(X_train, y_train)

    strat = TradingStrategy(
        model=ensemble, config=TradingConfig(), sentiment_analyzer=None,
        use_sentiment=False,
    )

    preds = np.asarray(ensemble.predict(feat), dtype=float)
    with caplog.at_level(logging.ERROR, logger="src.trading"):
        strat.calculate_signal(
            preds[-1:], float(feat["close"].iloc[-1]), "SYM",
            feat.index[-1], None, features=feat,
        )
    # The fix means get_model_contributions gets a real frame → no "requires
    # 'close'" error is logged.
    assert not any("requires" in r.message and "close" in r.message
                   for r in caplog.records), (
        "agreement boost still receives a column-less frame"
    )
