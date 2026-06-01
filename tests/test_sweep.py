"""Phase 2.7: tests for the hyperparameter-grid sweep + DSR wiring.

Split into:
  * Pure-helper unit tests (deterministic): grid expansion, equal-weight
    return combination, and the select-and-deflate summary (argmax selection,
    DSR ≤ PSR, NaN handling for <2 valid trials).
  * One end-to-end orchestration test: a 2-point xgboost-only static grid
    through ``run_sweep`` on a synthetic frame, asserting the trial loop runs,
    artifacts are written, and the DSR path executes when trials are valid.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest

from scripts.sweep import (
    TrialResult,
    combine_symbol_returns,
    expand_grid,
    run_sweep,
    summarize_sweep,
)
from src.validation.metrics import periodic_sharpe, probabilistic_sharpe_ratio


def _make_gbm_df(n: int, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.001, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    # tz-aware ET index (data_loader localizes bars; clean_data_for_training
    # asserts a tz-aware index, Phase 4 §4.2).
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


# ---------- expand_grid ----------


def test_expand_grid_cartesian_count() -> None:
    grid = {"a": [1, 2, 3], "b": [10, 20], "c": ["x"]}
    out = expand_grid(grid)
    assert len(out) == 3 * 2 * 1
    # Every combination present, keys preserved.
    assert {"a": 2, "b": 20, "c": "x"} in out


def test_expand_grid_default_18() -> None:
    from scripts.sweep import DEFAULT_GRID
    assert len(expand_grid(DEFAULT_GRID)) == 18


def test_expand_grid_deterministic_order() -> None:
    grid = {"a": [1, 2], "b": [3, 4]}
    out = expand_grid(grid)
    # dict-order axes, list-order values → reproducible trial indices.
    assert out == [
        {"a": 1, "b": 3}, {"a": 1, "b": 4},
        {"a": 2, "b": 3}, {"a": 2, "b": 4},
    ]


def test_expand_grid_empty_is_single_empty_config() -> None:
    assert expand_grid({}) == [{}]


# ---------- combine_symbol_returns ----------


def test_combine_single_symbol_passthrough() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    s = pd.Series([0.0, 0.01, -0.02, 0.03, 0.0], index=idx)
    out = combine_symbol_returns({"A": s})
    pd.testing.assert_series_equal(out, s)


def test_combine_two_symbols_equal_weight_by_date() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    a = pd.Series([0.02, 0.04, 0.06], index=idx)
    b = pd.Series([0.00, 0.00, 0.00], index=idx)
    out = combine_symbol_returns({"A": a, "B": b})
    assert out.tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_combine_partial_overlap_skipna() -> None:
    a = pd.Series([0.02, 0.04], index=pd.bdate_range("2024-01-02", periods=2))
    b = pd.Series([0.10], index=pd.bdate_range("2024-01-03", periods=1))
    out = combine_symbol_returns({"A": a, "B": b})
    # Day 1: only A → 0.02. Day 2: mean(0.04, 0.10) = 0.07.
    assert out.iloc[0] == pytest.approx(0.02)
    assert out.iloc[1] == pytest.approx(0.07)


def test_combine_empty_input() -> None:
    assert combine_symbol_returns({}).empty


# ---------- summarize_sweep ----------


def _trial(trial_id: int, mu: float, n: int = 1000, seed: int = 0) -> TrialResult:
    """A trial whose stored Sharpe is the periodic Sharpe of its returns."""
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(mu, 0.01, n))
    return TrialResult(
        trial_id=trial_id, config={"mu": mu},
        sharpe=float(periodic_sharpe(r)), combined_returns=r,
    )


def test_summarize_selects_argmax_and_deflates() -> None:
    # Increasing drift → increasing Sharpe; trial 3 should win.
    trials = [_trial(i, mu=0.0005 * i, seed=i) for i in range(4)]
    summary = summarize_sweep(trials)
    assert summary["n_trials"] == 4
    assert summary["n_valid"] == 4
    best = max(trials, key=lambda t: t.sharpe)
    assert summary["selected_trial_id"] == best.trial_id
    assert summary["selected_sharpe"] == pytest.approx(best.sharpe)
    # Deflation must penalize: DSR ≤ PSR(0) of the selected returns.
    psr = probabilistic_sharpe_ratio(best.combined_returns, sr_benchmark=0.0)
    assert summary["dsr"] is not None
    assert summary["dsr"] <= psr + 1e-9
    assert summary["sr_null"] is not None and summary["sr_null"] > 0.0


def test_summarize_single_valid_trial_dsr_none() -> None:
    trials = [_trial(0, mu=0.002, seed=1)]
    summary = summarize_sweep(trials)
    assert summary["n_valid"] == 1
    assert summary["selected_trial_id"] == 0  # still selected
    assert summary["dsr"] is None  # <2 trials → nothing to deflate
    assert summary["sr_null"] is None


def test_summarize_no_valid_trials() -> None:
    # Degenerate returns → NaN Sharpe → no valid trial.
    bad = TrialResult(
        trial_id=0, config={"mu": 0.0}, sharpe=float("nan"),
        combined_returns=pd.Series([0.01] * 10),
    )
    summary = summarize_sweep([bad])
    assert summary["n_valid"] == 0
    assert summary["selected_config"] is None
    assert summary["dsr"] is None
    assert summary["trial_sharpes"] == []


# ---------- end-to-end orchestration ----------


def test_run_sweep_end_to_end(monkeypatch, tmp_path) -> None:
    import src.tracking.mlflow_utils as mu
    from scripts.training import build_features
    from src.features import FeatureEngineer

    monkeypatch.setattr(mu, "MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    raw = _make_gbm_df(n=160, seed=11)
    fe = FeatureEngineer()
    full = build_features(
        raw_df=raw, symbol="SYM", feature_engineer=fe,
        sentiment_analyzer=None, horizon=5,
    )
    enhanced = {"SYM": full}

    args = SimpleNamespace(
        horizon=5, models="xgboost", timeframe="1d",
        start_date="2024-01-02", end_date=None,
        n_splits=2, embargo_pct=0.01, rolling=False, use_sentiment=False,
        initial_capital=10_000.0, position_size=0.2,
        stop_loss=0.02, take_profit=0.05,
        commission_bps=1.0, spread_bps=1.0, slippage_coeff=0.0,
        borrow_bps_annual=0.0, order_type="MOO",
        rl_timesteps=0, experiment="sweep_e2e_test",
    )
    # Aggressive vol target so positions clear the threshold and the strategy
    # actually trades (otherwise the equity curve is flat → NaN Sharpe).
    grid = {
        "target_vol": [3.0, 6.0],
        "signal_threshold": [0.05],
        "weighting_strategy": ["static"],
    }
    out_dir = tmp_path / "sweep_out"
    out_dir.mkdir()

    summary = run_sweep(args, enhanced, grid, out_dir)

    assert summary["n_trials"] == 2
    assert len(summary["trials"]) == 2
    assert (out_dir / "sweep_results.json").exists()
    assert (out_dir / "trial_sharpes.json").exists()
    # Per-trial training artifacts landed under trial_{i}/SYM/fold_*.
    assert (out_dir / "trial_0" / "SYM").is_dir()

    with open(out_dir / "trial_sharpes.json") as f:
        ts = json.load(f)["trial_sharpes"]
    assert len(ts) == summary["n_valid"]

    if summary["n_valid"] >= 2:
        configs = [t["config"] for t in summary["trials"]]
        assert summary["selected_config"] in configs
        assert summary["selected_sharpe"] == pytest.approx(
            max(t["sharpe"] for t in summary["trials"])
        )
        assert summary["dsr"] is not None and summary["psr"] is not None
        assert summary["dsr"] <= summary["psr"] + 1e-9
        assert summary["selected_config"] is not None
        assert (out_dir / "selected_config.json").exists()
