"""Tests for the PCA eigenportfolio factor model behind residual stat-arb."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.arbitrage.factors import (
    ResidualStatArbConfig,
    batched_factor_ols,
    compute_eligibility,
    compute_returns,
    consensus_trading_days,
    estimate_eigenportfolios,
    etf_factor_portfolios,
    volume_time_weights,
)


def _small_config(**overrides: object) -> ResidualStatArbConfig:
    base: dict = dict(
        corr_window=40,
        regr_window=15,
        n_factors=2,
        rebalance_every=5,
        min_price=5.0,
        min_median_dollar_volume=10_000.0,
        dollar_volume_window=5,
    )
    base.update(overrides)
    return ResidualStatArbConfig(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"corr_window": 10},
        {"regr_window": 60, "corr_window": 40},
        {"regr_window": 6, "n_factors": 2},
        {"s_entry_long": 0.5},
        {"s_exit_long": -2.0},
        {"s_exit_short": 2.0},
        {"max_half_life_bars": 0.0},
        {"position_unit": 0.0},
        {"rebalance_every": 0},
    ],
)
def test_config_rejects_bad_params(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _small_config(**overrides)


def test_compute_returns_never_pads_through_gaps() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="America/New_York")
    closes = pd.DataFrame({"AAA": [100.0, np.nan, 110.0, -5.0, 120.0]}, index=idx)
    returns = compute_returns(closes)
    assert np.isnan(returns.iloc[0, 0])
    assert np.isnan(returns.iloc[1, 0])  # missing close
    assert np.isnan(returns.iloc[2, 0])  # previous close missing: no synthetic return
    assert np.isnan(returns.iloc[3, 0])  # non-positive price treated as missing
    assert np.isnan(returns.iloc[4, 0])


def test_eligibility_floors_and_late_ipo() -> None:
    config = _small_config()
    n = 100
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="America/New_York")
    ok = np.full(n, 50.0)
    cheap = np.full(n, 3.0)  # below min_price forever
    late = np.full(n, 50.0)
    late[:30] = np.nan  # first finite return at bar 31
    thin = np.full(n, 50.0)  # liquid price, but volume floor fails
    closes = pd.DataFrame({"OK": ok, "CHEAP": cheap, "LATE": late, "THIN": thin}, index=idx)
    volumes = pd.DataFrame(
        {
            "OK": np.full(n, 1_000.0),
            "CHEAP": np.full(n, 1_000.0),
            "LATE": np.full(n, 1_000.0),
            "THIN": np.full(n, 1.0),
        },
        index=idx,
    )

    eligible = compute_eligibility(closes, volumes, config)
    # OK: first return at bar 1, so the 40-return window completes at bar 40.
    assert not eligible["OK"].iloc[39]
    assert eligible["OK"].iloc[40:].all()
    assert not eligible["CHEAP"].any()
    assert not eligible["THIN"].any()
    # LATE: first finite return at bar 31, full window at bar 70.
    assert not eligible["LATE"].iloc[:70].any()
    assert eligible["LATE"].iloc[70:].all()


def _liquid_panel(symbols: list[str], n: int = 80):
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="America/New_York")
    closes = pd.DataFrame({s: np.full(n, 50.0) for s in symbols}, index=idx)
    volumes = pd.DataFrame({s: np.full(n, 1_000.0) for s in symbols}, index=idx)
    return idx, closes, volumes


def test_eligibility_membership_mask_none_and_all_true_are_noops() -> None:
    config = _small_config()
    idx, closes, volumes = _liquid_panel(["A", "B"])
    base = compute_eligibility(closes, volumes, config)
    all_true = pd.DataFrame(True, index=idx, columns=["A", "B"])
    # None is the frozen-v1 path; an all-True mask must not change anything.
    pd.testing.assert_frame_equal(compute_eligibility(closes, volumes, config, None), base)
    pd.testing.assert_frame_equal(compute_eligibility(closes, volumes, config, all_true), base)


def test_eligibility_membership_mask_gates_and_defaults_nonmember_false() -> None:
    config = _small_config()
    idx, closes, volumes = _liquid_panel(["A", "B"])
    base = compute_eligibility(closes, volumes, config)
    assert base["A"].iloc[config.corr_window :].all()  # base-eligible once window fills

    # B leaves the index at bar 60: ineligible after, regardless of price/volume.
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    mask.iloc[60:, mask.columns.get_loc("B")] = False
    gated = compute_eligibility(closes, volumes, config, mask)
    assert gated["A"].equals(base["A"])  # A unaffected
    assert gated["B"].iloc[:60].equals(base["B"].iloc[:60])
    assert not gated["B"].iloc[60:].any()  # membership gate bites

    # A name absent from the mask columns is treated as a non-member (ineligible).
    gated2 = compute_eligibility(closes, volumes, config, pd.DataFrame(True, index=idx, columns=["A"]))
    assert not gated2["B"].any()
    assert gated2["A"].equals(base["A"])


def test_consensus_trading_days_drops_phantom_holiday_bars() -> None:
    idx = pd.date_range("2024-05-23", periods=6, freq="D", tz="America/New_York")
    # Row 3 is a phantom holiday: only one of fifty names carries a stray bar.
    data = {f"S{i}": [100.0, 101.0, 102.0, np.nan, 103.0, 104.0] for i in range(50)}
    data["S0"][3] = 102.5  # the lone vendor glitch bar
    closes = pd.DataFrame(data, index=idx)

    mask = consensus_trading_days(closes, min_fraction=0.6)
    assert mask.tolist() == [True, True, True, False, True, True]
    # A genuine session missing a couple of late-IPO names must survive.
    closes.iloc[1, :5] = np.nan  # 45/50 present on a real day
    assert consensus_trading_days(closes, min_fraction=0.6).iloc[1]


def test_consensus_trading_days_rejects_bad_fraction() -> None:
    closes = pd.DataFrame({"A": [1.0, 2.0]})
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            consensus_trading_days(closes, min_fraction=bad)


def test_volume_time_weights_direction_and_clip() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="America/New_York")
    vol = pd.DataFrame({"A": [1000.0] * 40}, index=idx)
    vol.iloc[35, 0] = 4000.0  # high-volume bar -> damp (weight < 1)
    vol.iloc[36, 0] = 250.0  # low-volume bar -> amplify (weight > 1), clips to 4
    w = volume_time_weights(vol, window=20, clip=4.0)
    assert w.iloc[34, 0] == pytest.approx(1.0, abs=0.05)  # typical bar ~ unity
    assert w.iloc[35, 0] < 1.0
    assert w.iloc[36, 0] == pytest.approx(4.0)  # ~4.45 ratio clipped to the band
    assert (w.to_numpy() >= 0.25 - 1e-9).all() and (w.to_numpy() <= 4.0 + 1e-9).all()


def test_volume_time_weights_causal_and_nan_safe() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="America/New_York")
    vol = pd.DataFrame({"A": [1000.0] * 40}, index=idx)
    base = volume_time_weights(vol, window=10, clip=4.0)
    future = vol.copy()
    future.iloc[30:] *= 10.0
    assert np.allclose(base.iloc[:30], volume_time_weights(future, window=10, clip=4.0).iloc[:30])
    # Non-positive / missing volume yields weight 1 (no information), not inf.
    vol.iloc[20, 0] = 0.0
    w = volume_time_weights(vol, window=10, clip=4.0)
    assert np.isfinite(w.to_numpy()).all()
    assert w.iloc[20, 0] == pytest.approx(1.0)


def test_volume_time_weights_rejects_bad_args() -> None:
    vol = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        volume_time_weights(vol, window=1, clip=4.0)
    with pytest.raises(ValueError):
        volume_time_weights(vol, window=10, clip=1.0)


def test_config_accepts_volume_time_and_rejects_bad_params() -> None:
    cfg = ResidualStatArbConfig(volume_time=True, volume_time_window=40, volume_time_clip=3.0)
    assert cfg.volume_time and cfg.volume_time_window == 40
    with pytest.raises(ValueError):
        ResidualStatArbConfig(volume_time_window=1)
    with pytest.raises(ValueError):
        ResidualStatArbConfig(volume_time_clip=1.0)


# ---------------------------------------------------------------------------
# ETF-factor mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"factor_mode": "nope"},  # unknown mode
        {"factor_mode": "etf"},  # etf mode needs factor symbols
        {"factor_mode": "etf", "etf_symbols": ("XLK", "XLK")},  # duplicate ETF
        {"factor_mode": "etf", "etf_symbols": tuple(f"E{i}" for i in range(12))},  # regr_window=15 < 12+5
        {"etf_symbols": ("XLK",)},  # etf_symbols supplied without etf mode
    ],
)
def test_config_rejects_bad_factor_mode(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _small_config(**overrides)


def test_config_accepts_etf_mode_and_normalizes_symbols() -> None:
    cfg = _small_config(factor_mode="etf", etf_symbols=("XLK", "xlf"))
    assert cfg.factor_mode == "etf"
    assert cfg.etf_symbols == ("XLK", "XLF")  # upper-cased and tuple-coerced
    assert cfg.n_model_factors == 2  # ETF count, not the (ignored) PCA n_factors
    assert _small_config().n_model_factors == _small_config().n_factors  # PCA default unchanged


def test_etf_factor_portfolios_select_unit_columns() -> None:
    symbols = ("AAA", "XLK", "BBB", "XLF")
    Q = etf_factor_portfolios(symbols, ("XLK", "XLF"))
    assert Q.shape == (2, 4)
    assert Q[0].tolist() == [0.0, 1.0, 0.0, 0.0]  # factor 0 = unit long XLK (col 1)
    assert Q[1].tolist() == [0.0, 0.0, 0.0, 1.0]  # factor 1 = unit long XLF (col 3)
    # Applying the selector to a return row extracts exactly the ETF returns,
    # so the PCA path's `window_R @ Q.T` yields the ETF factor returns verbatim.
    row = np.array([0.01, -0.02, 0.03, 0.04])
    assert (row @ Q.T).tolist() == [-0.02, 0.04]


def test_etf_factor_portfolios_rejects_absent_etf() -> None:
    with pytest.raises(ValueError, match="not a panel column"):
        etf_factor_portfolios(("AAA", "BBB"), ("XLK",))


def test_eligibility_marks_etf_factor_columns_ineligible() -> None:
    config = _small_config(factor_mode="etf", etf_symbols=("XLK",))
    n = 100
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="America/New_York")
    closes = pd.DataFrame({"OK": np.full(n, 50.0), "XLK": np.full(n, 200.0)}, index=idx)
    volumes = pd.DataFrame({"OK": np.full(n, 1_000.0), "XLK": np.full(n, 1e9)}, index=idx)
    eligible = compute_eligibility(closes, volumes, config)
    # XLK clears every price/liquidity floor but is a factor, never a stock leg.
    assert not eligible["XLK"].any()
    assert eligible["OK"].iloc[config.corr_window :].all()


def _one_factor_returns(n_obs: int = 300, n_assets: int = 6, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.01, size=n_obs)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    idio = rng.normal(0.0, 0.002, size=(n_obs, n_assets))
    return betas[None, :] * factor[:, None] + idio, factor


def test_eigenportfolios_recover_dominant_factor_with_positive_weights() -> None:
    returns, factor = _one_factor_returns()
    Q, explained = estimate_eigenportfolios(returns, n_factors=2)
    assert Q.shape == (2, returns.shape[1])
    assert explained[0] > 0.5
    # All loadings share the factor's sign, so the sign-fixed dominant
    # eigenportfolio must be all-positive and track the factor positively.
    assert (Q[0] > 0).all()
    factor_returns = returns @ Q.T
    corr = np.corrcoef(factor_returns[:, 0], factor)[0, 1]
    assert corr > 0.9


def test_eigenportfolios_deterministic() -> None:
    returns, _ = _one_factor_returns(seed=23)
    Q1, e1 = estimate_eigenportfolios(returns, n_factors=3)
    Q2, e2 = estimate_eigenportfolios(returns, n_factors=3)
    assert np.array_equal(Q1, Q2)
    assert np.array_equal(e1, e2)


def test_eigenportfolios_zero_variance_column_gets_zero_weight() -> None:
    returns, _ = _one_factor_returns()
    returns = returns.copy()
    returns[:, 2] = 0.0
    Q, explained = estimate_eigenportfolios(returns, n_factors=2)
    assert np.isfinite(Q).all()
    assert np.isfinite(explained).all()
    assert (Q[:, 2] == 0.0).all()


def test_eigenportfolios_reject_nan_window() -> None:
    returns, _ = _one_factor_returns()
    returns = returns.copy()
    returns[5, 1] = np.nan
    with pytest.raises(ValueError):
        estimate_eigenportfolios(returns, n_factors=2)


def test_batched_factor_ols_exact_recovery() -> None:
    rng = np.random.default_rng(3)
    factors = rng.normal(0.0, 0.01, size=(50, 2))
    alpha = np.array([0.001, -0.002, 0.0005])
    beta = np.array([[1.0, -0.5, 0.3], [0.2, 0.8, -1.1]])
    returns = alpha[None, :] + factors @ beta
    alpha_hat, beta_hat, residuals = batched_factor_ols(returns, factors)
    assert np.allclose(alpha_hat, alpha, atol=1e-12)
    assert np.allclose(beta_hat, beta, atol=1e-10)
    assert np.allclose(residuals, 0.0, atol=1e-12)


def test_batched_factor_ols_rejects_underdetermined_window() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        batched_factor_ols(rng.normal(size=(3, 2)), rng.normal(size=(3, 2)))
