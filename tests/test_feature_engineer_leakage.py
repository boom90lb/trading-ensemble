"""Regression tests for FeatureEngineer train-only outlier-clip contract.

The previous transform_features path recomputed mean/std from the dataframe
being transformed, then clipped values using those stats. When called on test
data this used test-set statistics — a row-level leak where a bar's clipped
value depended on later bars in the test window. Phase 2.8 audit (b) replaced
the recompute with stored fit-time bounds from fit_scalers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FeatureEngineer


def _ohlcv(seed: int, n: int = 200, mu: float = 0.0, sigma: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(mu, sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-03", periods=n, freq="D", tz="America/New_York")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_clip_bounds_recorded_after_fit():
    fe = FeatureEngineer()
    train = fe.create_features(_ohlcv(seed=0))
    fe.fit_scalers(train, "AAPL")
    bounds = fe.clip_bounds.get("AAPL", {})
    # Bounds should exist for the price/technical columns we clipped at fit.
    assert "close" in bounds
    lo, hi = bounds["close"]
    assert lo < hi
    # Bounds match mean±3σ on the subset fit_scalers actually fit on: rows where
    # every price-group column (open/high/low/close + moving averages) is non-NaN.
    price_cols = [c for c in train.columns if c.lower() in ["open", "high", "low", "close"] or c.startswith("ma") or c.startswith("ema")]
    fit_subset = train[price_cols].dropna()["close"]
    train_mean = float(fit_subset.mean())
    train_std = float(fit_subset.std())
    assert abs(lo - (train_mean - 3 * train_std)) < 1e-6
    assert abs(hi - (train_mean + 3 * train_std)) < 1e-6


def test_transform_does_not_depend_on_test_distribution():
    """The leak guard: prepending a giant outlier to test_b must not shift the
    clipped/scaled values of the rest of the rows that test_a and test_b share.
    """
    fe = FeatureEngineer()
    train = fe.create_features(_ohlcv(seed=0))
    fe.fit_scalers(train, "AAPL")

    test_base = fe.create_features(_ohlcv(seed=1))
    test_with_outlier = test_base.copy()
    # Inject a 100x-magnitude outlier at the front. If transform recomputed mean/std
    # over the whole df, every other row's clipped/standardized value would shift.
    test_with_outlier.iloc[0, test_with_outlier.columns.get_loc("close")] *= 100.0

    out_base = fe.transform_features(test_base, "AAPL", is_train=False)
    out_outlier = fe.transform_features(test_with_outlier, "AAPL", is_train=False)

    # All rows EXCEPT the injected one must be identical between the two transforms
    # (proving no row-level coupling via test-set statistics).
    np.testing.assert_array_equal(
        out_base["close"].iloc[1:].to_numpy(),
        out_outlier["close"].iloc[1:].to_numpy(),
    )


def test_test_outlier_clamped_to_train_bounds():
    """A test-time outlier should be clipped using train-fit bounds, not pass through
    or use its own distribution to set the bound."""
    fe = FeatureEngineer()
    train = fe.create_features(_ohlcv(seed=0))
    fe.fit_scalers(train, "AAPL")

    train_lo, train_hi = fe.clip_bounds["AAPL"]["close"]

    test_df = fe.create_features(_ohlcv(seed=1))
    extreme_value = train_hi * 10.0  # well above the train bound
    test_df.iloc[0, test_df.columns.get_loc("close")] = extreme_value

    out = fe.transform_features(test_df, "AAPL", is_train=False)
    # The scaler step shifts values; check the pre-scaler clip by replaying the
    # clip directly on the raw test value: the implementation must have clipped
    # to train_hi before scaling. We verify by feeding two test dfs where the
    # only difference is values beyond train_hi: their scaled outputs should match.
    test_df_2 = test_df.copy()
    test_df_2.iloc[0, test_df_2.columns.get_loc("close")] = train_hi * 50.0
    out_2 = fe.transform_features(test_df_2, "AAPL", is_train=False)
    # Both extreme values clip to train_hi, so the transformed row should be identical.
    assert out["close"].iloc[0] == out_2["close"].iloc[0]


def test_columns_not_seen_at_fit_pass_through():
    """A column that didn't exist at fit time (no stored bounds) should not be
    clipped at transform time — the previous code would recompute and clip it
    from the test df's own stats, which is the leak."""
    fe = FeatureEngineer()
    train = fe.create_features(_ohlcv(seed=0))
    fe.fit_scalers(train, "AAPL")

    test_df = fe.create_features(_ohlcv(seed=1))
    # Drop a column from the FIT set and add a brand-new numeric column at transform.
    test_df["never_fit_col"] = np.linspace(-100.0, 100.0, len(test_df))

    out = fe.transform_features(test_df, "AAPL", is_train=False)
    # The new column must pass through unchanged (no stored bound, no recompute).
    np.testing.assert_array_equal(out["never_fit_col"].to_numpy(), test_df["never_fit_col"].to_numpy())
