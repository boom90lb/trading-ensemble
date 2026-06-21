"""Offline smoke tests for script-level research claim packets."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.arbitrage import PairCandidate
from src.execution.target_weights import backtest_target_weights
from src.models.base import BaseModel
from src.validation.trials import validate_claim_packet_dir


class _DummyRun:
    def __enter__(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="dummy"))

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_mlflow(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "init_mlflow", lambda *a, **k: "0", raising=False)
    monkeypatch.setattr(module.mlflow, "start_run", lambda *a, **k: _DummyRun(), raising=False)
    monkeypatch.setattr(module, "log_metrics_safe", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module, "log_params_safe", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module, "log_artifact_dir", lambda *a, **k: None, raising=False)


def _bars(n: int = 64) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=n, tz="America/New_York")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _feature_frame(raw_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = raw_df.copy()
    out[f"target_{horizon}"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return out


class _ConstModel(BaseModel):
    def __init__(self, value: float = 0.6):
        super().__init__(name="const", target_column="close", horizon=1)
        self.value = value
        self.is_fitted = True

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)

    def save(self, directory: Path) -> Path:
        return Path(directory)

    def load(self, model_path: Path):
        return self

    def evaluate(self, X_test, y_test):
        return {}


def test_stat_arb_main_emits_valid_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import scripts.stat_arb as mod

    close = pd.DataFrame({"AAA": _bars()["close"], "BBB": _bars()["close"] * 1.01})
    open_ = pd.DataFrame({"AAA": _bars()["open"], "BBB": _bars()["open"] * 1.01})
    candidate = PairCandidate(
        asset_y="AAA",
        asset_x="BBB",
        alpha=0.0,
        beta=1.0,
        coint_pvalue=0.01,
        coint_stat=-4.0,
        adf_pvalue=0.01,
        adf_stat=-4.0,
        half_life=5.0,
        beta_drift=0.1,
        return_corr=0.9,
        n_obs=20,
    )

    monkeypatch.setattr(mod, "_fetch_matrices", lambda *a, **k: (close, open_))
    monkeypatch.setattr(mod, "scan_cointegrated_pairs", lambda *a, **k: [candidate])
    monkeypatch.setattr(
        mod,
        "generate_pair_positions",
        lambda prices, _candidate, _cfg: SimpleNamespace(
            target_weights=pd.DataFrame(
                {"AAA": 0.5, "BBB": -0.5},
                index=prices.index,
            )
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "stat_arb.py",
        "--symbols",
        "AAA,BBB",
        "--formation_bars",
        "30",
        "--output_dir",
        str(tmp_path),
    ])

    mod.main()

    packet = validate_claim_packet_dir(tmp_path)
    assert packet["strategy"] == "pairs_stat_arb"


def test_stat_arb_wfo_main_emits_valid_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import scripts.stat_arb_wfo as mod

    close = pd.DataFrame({"AAA": _bars()["close"], "BBB": _bars()["close"] * 0.99})
    open_ = pd.DataFrame({"AAA": _bars()["open"], "BBB": _bars()["open"] * 0.99})
    targets = pd.DataFrame({"AAA": 0.4, "BBB": -0.4}, index=open_.index)
    portfolio = backtest_target_weights(open_, targets)
    fake_fold = object()
    fake_result = SimpleNamespace(
        folds=(fake_fold,),
        portfolio=portfolio,
        pair_trial_sharpes=(0.1, 0.2),
        summary=portfolio.metrics | {"pair_set_dsr": 0.1},
    )

    monkeypatch.setattr(mod, "_fetch_matrices", lambda *a, **k: (close, open_))
    monkeypatch.setattr(mod, "run_stat_arb_walk_forward", lambda *a, **k: fake_result)
    monkeypatch.setattr(
        mod,
        "fold_result_to_dict",
        lambda _fold: {
            "fold": 0,
            "formation_start": close.index[0].isoformat(),
            "formation_end": close.index[9].isoformat(),
            "test_start": close.index[10].isoformat(),
            "test_end": close.index[-1].isoformat(),
            "selected_pairs": [],
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "stat_arb_wfo.py",
        "--symbols",
        "AAA,BBB",
        "--formation_bars",
        "30",
        "--test_bars",
        "20",
        "--output_dir",
        str(tmp_path),
    ])

    mod.main()

    packet = validate_claim_packet_dir(tmp_path)
    assert packet["strategy"] == "pairs_stat_arb_wfo"


def test_backtest_main_target_weights_emits_root_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import scripts.backtest as mod

    _patch_mlflow(monkeypatch, mod)
    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    raw = _bars()
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": False,
        "models": ["xgboost"],
        "n_splits": 1,
        "purge_horizon": 1,
        "embargo_pct": 0.0,
        "expanding": True,
    }

    class _Loader:
        def fetch_historical_data(self, *args, **kwargs):
            return raw

        def fetch_dividends(self, *args, **kwargs):
            return pd.Series([1.0], index=[raw.index[5]], name="amount")

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        mod,
        "build_features",
        lambda raw_df, symbol, feature_engineer, sentiment_analyzer, horizon: _feature_frame(raw_df, horizon),
    )
    monkeypatch.setattr(mod, "load_fold_ensemble", lambda *a, **k: _ConstModel(0.7))
    monkeypatch.setattr(
        mod,
        "_fold_date_range",
        lambda _fold, usable, **kwargs: (usable.index[0], usable.index[-1]),
    )
    monkeypatch.setattr(mod, "_fold_model_data", lambda _fold, symbol, full_df: full_df)
    monkeypatch.setattr(sys, "argv", [
        "backtest.py",
        "--training_run",
        str(training_run),
        "--symbols",
        "AAA",
    ])

    mod.main()

    out_dirs = sorted(tmp_path.glob("wfo_backtest_*"))
    assert out_dirs
    packet = validate_claim_packet_dir(out_dirs[-1])
    assert packet["strategy"] == "ensemble_wfo_target_weights"
    assert (out_dirs[-1] / "target_weights.csv").exists()
    assert (out_dirs[-1] / "fill_weights.csv").exists()


def test_rl_seed_eval_main_emits_member_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import scripts.rl_seed_eval as mod

    _patch_mlflow(monkeypatch, mod)
    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    raw = _bars()
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": False,
    }

    class _Loader:
        def fetch_historical_data(self, *args, **kwargs):
            return raw

        def fetch_dividends(self, *args, **kwargs):
            return pd.Series(dtype=float)

    def _seed_returns(member, seed, **kwargs):
        idx = pd.bdate_range("2026-03-02", periods=30)
        values = 0.001 + seed * 0.0001 + np.linspace(-0.0005, 0.0005, len(idx))
        return pd.Series(values, index=idx, name=member)

    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        mod,
        "build_features",
        lambda raw_df, symbol, feature_engineer, sentiment_analyzer, horizon: _feature_frame(raw_df, horizon),
    )
    monkeypatch.setattr(mod, "eval_member_seed", _seed_returns)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "rl_seed_eval.py",
        "--training_run",
        str(training_run),
        "--members",
        "lstm_ppo",
        "--seeds",
        "0,1",
        "--output_dir",
        str(out_dir),
    ])

    mod.main()

    packet = validate_claim_packet_dir(out_dir / "lstm_ppo")
    assert packet["strategy"] == "rl_member_seed_robustness"


def test_rl_seed_eval_rejects_sentiment_before_data_loading(monkeypatch, tmp_path: Path) -> None:
    import scripts.rl_seed_eval as mod

    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": True,
    }

    class _Loader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("DataLoader should not be constructed")

    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(sys, "argv", [
        "rl_seed_eval.py",
        "--training_run",
        str(training_run),
        "--members",
        "lstm_ppo",
        "--seeds",
        "0",
        "--output_dir",
        str(tmp_path / "out"),
    ])

    with pytest.raises(ValueError, match="sentiment is disabled for rl_seed_eval"):
        mod.main()
