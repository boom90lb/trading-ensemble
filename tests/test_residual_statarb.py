"""Tests for the residual (Avellaneda-Lee) stat-arb signal path and walk-forward."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.stat_arb_residual_wfo import _append_trial, _config_hash, _load_trial_sharpes
from src.arbitrage.factors import ResidualStatArbConfig, etf_factor_portfolios
from src.arbitrage.residual import (
    OUFit,
    apply_no_trade_band,
    build_book_row,
    cap_book,
    compute_residual_signal_panel,
    cost_aware_band,
    fit_ou_batch,
    next_states,
    run_state_machine,
    sscore_batch,
)
from src.arbitrage.residual_walk_forward import (
    residual_fold_to_dict,
    run_residual_stat_arb_walk_forward,
)
from src.arbitrage.walk_forward import StatArbWalkForwardConfig
from src.config import ExecutionConfig

_FROZEN = ResidualStatArbConfig()


def _small_config(**overrides: object) -> ResidualStatArbConfig:
    base: dict = dict(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        rebalance_every=5,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )
    base.update(overrides)
    return ResidualStatArbConfig(**base)


def _synthetic_panel(
    n_days: int = 240, n_assets: int = 8, seed: int = 7, resid_b: float = 0.55, resid_vol: float = 0.012
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor + AR(1)-residual returns, so residual signals genuinely exist."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="America/New_York")
    factor = rng.normal(0.0003, 0.01, size=n_days)
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    levels = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        levels[t] = resid_b * levels[t - 1] + rng.normal(0.0, resid_vol, size=n_assets)
    residual_returns = np.diff(levels, axis=0, prepend=np.zeros((1, n_assets)))
    returns = betas[None, :] * factor[:, None] + residual_returns
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=idx,
        columns=[f"S{i}" for i in range(n_assets)],
    )
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=closes.columns)
    return closes, volumes


# ---------------------------------------------------------------------------
# OU fit and s-score
# ---------------------------------------------------------------------------


def test_fit_ou_batch_recovers_ar1_parameters() -> None:
    rng = np.random.default_rng(42)
    n = 800
    b_true, a_true, sigma_true = 0.7, 0.3, 0.05
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = a_true + b_true * X[t - 1] + rng.normal(0.0, sigma_true)
    eps = np.diff(X, prepend=0.0)  # cumsum(eps) reconstructs X exactly

    fit = fit_ou_batch(eps[:, None])
    assert fit.valid[0]
    assert fit.b[0] == pytest.approx(b_true, abs=0.05)
    assert fit.m[0] == pytest.approx(a_true / (1 - b_true), abs=0.15)
    assert fit.sigma_eq[0] == pytest.approx(sigma_true / math.sqrt(1 - b_true**2), rel=0.2)
    assert fit.half_life_bars[0] == pytest.approx(-math.log(2) / math.log(b_true), abs=0.5)


def test_fit_ou_batch_flags_explosive_and_degenerate() -> None:
    rng = np.random.default_rng(1)
    n = 120
    explosive = 1.05 ** np.arange(n)
    flat = np.zeros(n)
    ok = np.zeros(n)
    for t in range(1, n):
        ok[t] = 0.6 * ok[t - 1] + rng.normal(0.0, 0.02)
    X = np.column_stack([explosive, flat, ok])
    eps = np.diff(X, axis=0, prepend=np.zeros((1, 3)))

    fit = fit_ou_batch(eps)
    assert not fit.valid[0]  # b >= 1: not mean-reverting
    assert not fit.valid[1]  # zero variance: degenerate
    assert fit.valid[2]


def test_sscore_drift_adjustment_shifts_center() -> None:
    fit = OUFit(
        b=np.array([0.5]),
        m=np.array([0.0]),
        sigma_eq=np.array([1.0]),
        half_life_bars=np.array([1.0]),
        x_end=np.array([0.0]),
        valid=np.array([True]),
    )
    neutral = sscore_batch(fit, np.array([0.0]))[0]
    positive_drift = sscore_batch(fit, np.array([0.01]))[0]
    assert neutral == pytest.approx(0.0)
    # Positive idiosyncratic drift raises the equilibrium level, lowering s.
    assert positive_drift == pytest.approx(-0.01 / math.log(2), rel=1e-6)

    invalid = OUFit(
        b=np.array([1.2]),
        m=np.array([0.0]),
        sigma_eq=np.array([1.0]),
        half_life_bars=np.array([np.inf]),
        x_end=np.array([0.0]),
        valid=np.array([False]),
    )
    assert np.isnan(sscore_batch(invalid, np.array([0.0]))[0])


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "s", "ok", "expected"),
    [
        (0, -1.30, True, 1),  # open long below entry band
        (0, -1.20, True, 0),  # inside band: stay flat
        (0, 1.30, True, -1),  # open short above entry band
        (0, 0.0, True, 0),
        (1, -0.60, True, 1),  # still below the -0.50 close band: hold
        (1, -0.40, True, 0),  # crossed the close band: exit
        (1, 2.00, True, -1),  # exit and flip in one evaluation
        (-1, 0.80, True, -1),  # still above the +0.75 close band: hold
        (-1, 0.70, True, 0),  # crossed the close band: exit
        (-1, -2.00, True, 1),  # exit and flip
        (1, -2.00, False, 0),  # invalid signal forces flat regardless of s
        (-1, float("nan"), False, 0),
        (0, float("nan"), False, 0),
    ],
)
def test_next_states_transition_table(state: int, s: float, ok: bool, expected: int) -> None:
    out = next_states(np.array([state], dtype=np.int8), np.array([s]), np.array([ok]), _FROZEN)
    assert out[0] == expected


def test_run_state_machine_requires_fresh_crossing_to_reenter() -> None:
    s = np.array([[-1.3], [-0.8], [-0.3], [-0.8]])
    ok = np.ones_like(s, dtype=bool)
    states = run_state_machine(s, ok, _FROZEN)
    assert states[:, 0].tolist() == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# Book construction and caps
# ---------------------------------------------------------------------------


def test_build_book_row_nets_stock_and_hedge_legs_per_symbol() -> None:
    states = np.array([1, 1, 0], dtype=np.int8)
    beta_day = np.array([[0.5, 1.0, 0.0]])  # (k=1, n_symbols=3)
    eigenportfolios = np.array([[0.2, -0.1, 0.3]])
    row = build_book_row(states, beta_day, eigenportfolios, position_unit=1.0)
    # Net factor exposure 1.5 hedged through the eigenportfolio, netted with
    # the stock legs BEFORE any cap: [1-0.3, 1+0.15, -0.45].
    assert np.allclose(row, [0.7, 1.15, -0.45])


def test_cap_book_scales_gross_proportionally_then_clips_symbols() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="B")
    targets = pd.DataFrame({"AAA": [1.2, 0.1], "BBB": [-0.8, -0.1]}, index=idx)
    capped = cap_book(targets, max_gross=1.0, max_symbol_abs_weight=0.35)
    # Row 0 gross 2.0 -> scaled to [0.6, -0.4] (ratios preserved), then clipped.
    assert capped.iloc[0].tolist() == [0.35, -0.35]
    # Row 1 is under both caps and must be untouched.
    assert capped.iloc[1].tolist() == [0.1, -0.1]
    assert (capped.abs().sum(axis=1) <= 1.0 + 1e-12).all()


# ---------------------------------------------------------------------------
# Signal panel: causality and eligibility
# ---------------------------------------------------------------------------


def test_signal_panel_is_causal_under_future_perturbation() -> None:
    closes, volumes = _synthetic_panel()
    config = _small_config()
    cut = 200

    perturbed = closes.copy()
    rng = np.random.default_rng(99)
    perturbed.iloc[cut:] = perturbed.iloc[cut:].to_numpy() * np.exp(
        rng.normal(0.0, 0.05, size=perturbed.iloc[cut:].shape)
    )
    base = compute_residual_signal_panel(closes, volumes, config)
    other = compute_residual_signal_panel(perturbed, volumes, config)

    assert np.array_equal(base.q_index[:cut], other.q_index[:cut])
    assert base.active.iloc[:cut].equals(other.active.iloc[:cut])
    assert base.tradeable.iloc[:cut].equals(other.tradeable.iloc[:cut])
    assert np.allclose(base.sscore.iloc[:cut], other.sscore.iloc[:cut], equal_nan=True)
    assert np.allclose(base.beta[:cut], other.beta[:cut])


def test_signal_panel_eligibility_loss_stops_signals() -> None:
    closes, volumes = _synthetic_panel()
    config = _small_config()
    drop_at = 150
    closes = closes.copy()
    closes.iloc[drop_at:, 0] = 3.0  # below min_price -> ineligible from drop_at on

    panel = compute_residual_signal_panel(closes, volumes, config)
    symbol = closes.columns[0]
    assert panel.active[symbol].iloc[config.warmup_bars : drop_at].any()
    assert not panel.active[symbol].iloc[drop_at:].any()
    assert not panel.tradeable[symbol].iloc[drop_at:].any()


# ---------------------------------------------------------------------------
# Volume "trading time" (A-L §6 ablation)
# ---------------------------------------------------------------------------


def test_volume_time_off_matches_frozen_and_on_changes_signal() -> None:
    closes, volumes = _synthetic_panel(seed=5)
    rng = np.random.default_rng(0)
    volumes = volumes * rng.uniform(0.3, 3.0, size=volumes.shape)  # atypical volume so weighting bites

    base = compute_residual_signal_panel(closes, volumes, _small_config())
    off = compute_residual_signal_panel(closes, volumes, _small_config(volume_time=False))
    on = compute_residual_signal_panel(closes, volumes, _small_config(volume_time=True, volume_time_window=40))

    # Default and explicit-off are the frozen path, bit-for-bit.
    assert np.array_equal(off.sscore.to_numpy(), base.sscore.to_numpy(), equal_nan=True)
    # Trading time perturbs s-scores where volume is atypical, but keeps the
    # same active set (eligibility is on raw returns, unchanged).
    a, b = off.sscore.to_numpy(), on.sscore.to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0
    assert not np.allclose(a[both], b[both])
    assert off.active.equals(on.active)


# ---------------------------------------------------------------------------
# ETF-factor mode
# ---------------------------------------------------------------------------


def _synthetic_etf_panel(
    n_days: int = 240, n_stocks: int = 6, seed: int = 11, resid_b: float = 0.55, resid_vol: float = 0.012
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Stocks loading on two observable sector-ETF columns plus AR(1) residuals.

    Prices are exact cumulative products of the intended simple returns, so the
    ETF columns' recovered returns are the factors the OLS regresses on.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="America/New_York")
    etf_returns = rng.normal(0.0003, 0.01, size=(n_days, 2))
    loadings = rng.uniform(0.4, 1.3, size=(2, n_stocks))
    levels = np.zeros((n_days, n_stocks))
    for t in range(1, n_days):
        levels[t] = resid_b * levels[t - 1] + rng.normal(0.0, resid_vol, size=n_stocks)
    residual_returns = np.diff(levels, axis=0, prepend=np.zeros((1, n_stocks)))
    stock_returns = etf_returns @ loadings + residual_returns
    etf_cols = ["XLK", "XLF"]
    closes = pd.DataFrame(
        np.column_stack(
            [100.0 * np.cumprod(1.0 + stock_returns, axis=0), 100.0 * np.cumprod(1.0 + etf_returns, axis=0)]
        ),
        index=idx,
        columns=[f"S{i}" for i in range(n_stocks)] + etf_cols,
    )
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=closes.columns)
    return closes, volumes, tuple(etf_cols)


def _etf_config(**overrides: object) -> ResidualStatArbConfig:
    base: dict = dict(factor_mode="etf", etf_symbols=("XLK", "XLF"))
    base.update(overrides)
    return _small_config(**base)


def test_build_book_row_hedges_onto_etf_columns() -> None:
    states = np.array([1, -1, 0, 0], dtype=np.int8)  # long S0, short S1; ETFs flat
    beta_day = np.array([[0.8, 0.5, 0.0, 0.0], [0.2, 0.9, 0.0, 0.0]])  # rows = XLK, XLF loadings
    Q = etf_factor_portfolios(("S0", "S1", "XLK", "XLF"), ("XLK", "XLF"))
    row = build_book_row(states, beta_day, Q, position_unit=1.0)
    # Net factor exposure [0.8-0.5, 0.2-0.9] = [0.3, -0.7] hedged on the ETF columns.
    assert np.allclose(row, [1.0, -1.0, -0.3, 0.7])
    # The resulting book is factor-neutral: stock betas + ETF self-exposure net to zero.
    assert np.allclose(beta_day @ row + row[2:], 0.0)


def test_etf_panel_excludes_etf_columns_and_uses_fixed_selector() -> None:
    closes, volumes, etf_cols = _synthetic_etf_panel()
    config = _etf_config()
    panel = compute_residual_signal_panel(closes, volumes, config)
    # ETF mode has one fixed factor portfolio, pointed to from warmup onward.
    assert len(panel.eigenportfolios) == 1
    assert panel.beta.shape[1] == 2  # n_model_factors == number of ETF factors
    assert (panel.q_index[config.warmup_bars :] == 0).all()
    assert np.array_equal(panel.eigenportfolios[0], etf_factor_portfolios(panel.symbols, config.etf_symbols))
    # ETF columns are factors/hedges only: never active, never tradeable.
    for etf in etf_cols:
        assert not panel.active[etf].any()
        assert not panel.tradeable[etf].any()
    stock_cols = [c for c in closes.columns if c not in etf_cols]
    assert panel.active[stock_cols].to_numpy().any()
    assert panel.tradeable[stock_cols].to_numpy().any()


def test_etf_panel_is_causal_under_future_perturbation() -> None:
    closes, volumes, etf_cols = _synthetic_etf_panel()
    config = _etf_config()
    cut = 200
    stock_cols = [c for c in closes.columns if c not in etf_cols]
    perturbed = closes.copy()
    rng = np.random.default_rng(99)
    block = perturbed.loc[perturbed.index[cut:], stock_cols].to_numpy()
    perturbed.loc[perturbed.index[cut:], stock_cols] = block * np.exp(rng.normal(0.0, 0.05, size=block.shape))

    base = compute_residual_signal_panel(closes, volumes, config)
    other = compute_residual_signal_panel(perturbed, volumes, config)
    assert base.active.iloc[:cut].equals(other.active.iloc[:cut])
    assert np.allclose(base.sscore.iloc[:cut], other.sscore.iloc[:cut], equal_nan=True)
    assert np.allclose(base.beta[:cut], other.beta[:cut])


def test_etf_wfo_smoke_hedges_on_etf_columns() -> None:
    closes, volumes, etf_cols = _synthetic_etf_panel()
    result = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        signal_config=_etf_config(),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=40, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )
    assert len(result.folds) >= 1
    assert result.summary["n_factor_etfs"] == 2.0
    # ETFs are excluded from the tradeable-name count but still priced/held.
    assert result.summary["n_symbols"] == len(closes.columns) - 2
    tw = result.portfolio.target_weights
    assert tw[list(etf_cols)].abs().to_numpy().sum() > 0  # hedge legs land on ETFs
    stock_cols = [c for c in closes.columns if c not in etf_cols]
    assert tw[stock_cols].abs().to_numpy().sum() > 0  # stock legs trade
    assert result.summary["avg_names_traded"] > 0
    assert (result.portfolio.costs["gross"] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def _run_wfo(closes: pd.DataFrame, volumes: pd.DataFrame, **walk_overrides: object):
    walk: dict = dict(formation_bars=80, test_bars=40, min_test_bars=20)
    walk.update(walk_overrides)
    return run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),  # open == close keeps the test panel simple
        volumes,
        signal_config=_small_config(),
        walk_config=StatArbWalkForwardConfig(**walk),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )


def test_no_trade_band_holds_until_target_moves() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    tgt = pd.DataFrame({"A": [0.10, 0.12, 0.09, 0.40, 0.41]}, index=idx)
    out = apply_no_trade_band(tgt, 0.05)
    # snap to 0.10, hold through the small wobbles, snap to 0.40 on the big move, hold.
    assert list(out["A"]) == [0.10, 0.10, 0.10, 0.40, 0.40]


def test_no_trade_band_zero_is_strict_noop() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    tgt = pd.DataFrame({"A": [0.1, -0.2, 0.3, 0.0], "B": [0.0, 0.1, -0.1, 0.2]}, index=idx)
    pd.testing.assert_frame_equal(apply_no_trade_band(tgt, 0.0), tgt)


def test_walk_config_rejects_negative_no_trade_band() -> None:
    with pytest.raises(ValueError):
        StatArbWalkForwardConfig(no_trade_band=-0.1)


def test_no_trade_band_cuts_turnover_in_wfo() -> None:
    closes, volumes = _synthetic_panel()
    base = _run_wfo(closes, volumes)
    banded = _run_wfo(closes, volumes, no_trade_band=0.02)
    assert banded.summary["avg_turnover"] <= base.summary["avg_turnover"]


def test_cost_aware_band_monotonicity() -> None:
    # Slower reversion (longer half-life) => wider band.
    short_hl = cost_aware_band(np.array([5.0]), per_trade_cost_frac=4e-4)[0]
    long_hl = cost_aware_band(np.array([50.0]), per_trade_cost_frac=4e-4)[0]
    assert long_hl > short_hl
    # Higher round-trip cost => wider band.
    low_cost = cost_aware_band(np.array([20.0]), per_trade_cost_frac=1e-4)[0]
    high_cost = cost_aware_band(np.array([20.0]), per_trade_cost_frac=1e-3)[0]
    assert high_cost > low_cost


def test_cost_aware_band_degenerate_is_zero() -> None:
    # NaN / inf / non-positive half-lives carry no OU speed => band 0 (no extra hold).
    bands = cost_aware_band(np.array([np.nan, np.inf, 0.0, -3.0]), per_trade_cost_frac=4e-4)
    assert np.array_equal(bands, np.zeros(4))


def test_per_name_band_series_applies() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    path = [0.10, 0.12, 0.09, 0.13, 0.10, 0.14]  # small wobbles, max deviation 0.04
    targets = pd.DataFrame({"HI": path, "LO": path}, index=idx)
    band = pd.Series({"HI": 0.10, "LO": 0.0})  # HI's wide band absorbs the wobble; LO trades every move
    out = apply_no_trade_band(targets, band)
    assert out["HI"].diff().abs().sum() < out["LO"].diff().abs().sum()


def test_band_mode_fixed_is_parity() -> None:
    closes, volumes = _synthetic_panel()
    default = _run_wfo(closes, volumes)  # band_mode defaults to "fixed", no_trade_band 0
    explicit = _run_wfo(closes, volumes, band_mode="fixed", no_trade_band=0.0)
    assert default.summary == explicit.summary
    # cost_aware mode runs end-to-end through the new per-name path (finite summary).
    cost_aware = _run_wfo(closes, volumes, band_mode="cost_aware")
    assert math.isfinite(cost_aware.summary["oos_periodic_sharpe"])


def test_build_book_row_size_scale_none_parity() -> None:
    states = np.array([1, -1, 0], dtype=np.int8)
    beta_day = np.array([[0.5, 1.0, 0.0]])
    eig = np.array([[0.2, -0.1, 0.3]])
    base = build_book_row(states, beta_day, eig, position_unit=0.02)
    explicit = build_book_row(states, beta_day, eig, position_unit=0.02, size_scale=None)
    assert np.array_equal(base, explicit)


def test_build_book_row_scales_legs() -> None:
    states = np.array([1, 1], dtype=np.int8)
    beta_day = np.zeros((1, 2))  # no factor exposure => no hedge => row is the stock legs
    eig = np.zeros((1, 2))
    unit = build_book_row(states, beta_day, eig, position_unit=0.02, size_scale=np.array([1.0, 1.0]))
    scaled = build_book_row(states, beta_day, eig, position_unit=0.02, size_scale=np.array([2.0, 1.0]))
    assert scaled[0] == pytest.approx(2.0 * unit[0])
    assert scaled[1] == pytest.approx(unit[1])


def test_sizing_mode_unit_is_parity() -> None:
    closes, volumes = _synthetic_panel()
    default = _run_wfo(closes, volumes)  # _small_config() => sizing_mode default "unit"
    explicit = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        signal_config=_small_config(sizing_mode="unit"),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=40, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )
    assert default.summary == explicit.summary


def test_strength_sizing_concentrates_gross() -> None:
    closes, volumes = _synthetic_panel()
    common = dict(
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=40, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )
    unit = run_residual_stat_arb_walk_forward(
        closes, closes.copy(), volumes, signal_config=_small_config(sizing_mode="unit"), **common
    )
    strength = run_residual_stat_arb_walk_forward(
        closes, closes.copy(), volumes, signal_config=_small_config(sizing_mode="strength"), **common
    )
    panel = compute_residual_signal_panel(closes, volumes, _small_config())  # sscore is sizing-independent

    def gross_weighted_abs_s(result) -> float:
        tw = result.portfolio.target_weights
        a = np.abs(panel.sscore.reindex(index=tw.index, columns=tw.columns).to_numpy())
        w = np.abs(tw.to_numpy())
        m = (w > 0) & np.isfinite(a)
        return float((w[m] * a[m]).sum() / w[m].sum())

    # Strength sizing redistributes the same gross budget toward higher-|s| names.
    assert gross_weighted_abs_s(strength) > gross_weighted_abs_s(unit)


def test_residual_wfo_smoke_trades_and_reports() -> None:
    closes, volumes = _synthetic_panel()
    result = _run_wfo(closes, volumes)
    assert len(result.folds) == 4
    for key in (
        "sharpe",
        "n_folds",
        "n_symbols",
        "avg_names_traded",
        "signal_evaluations",
        "invalid_ou_rate",
        "slow_ou_rate",
        "n_rebalances",
        "oos_periodic_sharpe",
    ):
        assert key in result.summary
    assert result.summary["signal_evaluations"] > 0
    assert result.summary["avg_names_traded"] > 0
    assert result.summary["avg_turnover"] > 0
    assert 0.0 <= result.summary["invalid_ou_rate"] <= 1.0
    assert (result.portfolio.costs["gross"] <= 1.0 + 1e-9).all()


def test_residual_wfo_passes_dollar_volume_to_adv_impact() -> None:
    closes, volumes = _synthetic_panel()
    common = dict(
        signal_config=_small_config(),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=40, min_test_bars=20),
    )
    base = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
        **common,
    )
    capacity = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        execution=ExecutionConfig(
            spread_bps=0,
            commission_bps=0,
            slippage_coeff=0,
            borrow_rate_bps_annual=0,
            adv_impact_coeff=1000,
            adv_floor_dollars=1.0,
        ),
        **common,
    )
    assert capacity.portfolio.costs["impact"].sum() > base.portfolio.costs["impact"].sum()
    assert capacity.portfolio.costs["total"].sum() > base.portfolio.costs["total"].sum()


def test_residual_wfo_flattens_fold_boundaries() -> None:
    closes, volumes = _synthetic_panel()
    result = _run_wfo(closes, volumes)
    for fold in result.folds:
        assert result.portfolio.target_weights.loc[fold.test_end].abs().sum() == 0.0
    for previous, current in zip(result.folds, result.folds[1:]):
        assert (
            result.portfolio.fill_weights.loc[current.test_start].abs().sum() == 0.0
        ), f"carry from fold {previous.fold} into fold {current.fold}"


def test_residual_wfo_deterministic() -> None:
    closes, volumes = _synthetic_panel()
    first = _run_wfo(closes, volumes)
    second = _run_wfo(closes, volumes)
    assert [residual_fold_to_dict(f) for f in first.folds] == [residual_fold_to_dict(f) for f in second.folds]
    assert first.summary == second.summary


def test_residual_wfo_rejects_formation_below_warmup() -> None:
    closes, volumes = _synthetic_panel()
    with pytest.raises(ValueError, match="warmup"):
        _run_wfo(closes, volumes, formation_bars=79)


# ---------------------------------------------------------------------------
# Trial ledger
# ---------------------------------------------------------------------------


def test_trial_ledger_roundtrip_and_corruption_handling(tmp_path) -> None:
    ledger = tmp_path / "trials.jsonl"
    _append_trial(ledger, {"oos_periodic_sharpe": 0.05, "config_hash": "a"})
    _append_trial(ledger, {"oos_periodic_sharpe": float("nan"), "config_hash": "b"})
    _append_trial(ledger, {"oos_periodic_sharpe": -0.01, "config_hash": "c"})
    with ledger.open("a") as f:
        f.write("{not json}\n")

    sharpes = _load_trial_sharpes(ledger)
    assert sharpes == [0.05, -0.01]
    assert _load_trial_sharpes(tmp_path / "missing.jsonl") == []


def test_config_hash_is_stable_and_sensitive() -> None:
    payload = {"signal": {"corr_window": 252, "n_factors": 15}, "symbols": ["A", "B"]}
    reordered = {"symbols": ["A", "B"], "signal": {"n_factors": 15, "corr_window": 252}}
    assert _config_hash(payload) == _config_hash(reordered)
    changed = {"signal": {"corr_window": 252, "n_factors": 14}, "symbols": ["A", "B"]}
    assert _config_hash(payload) != _config_hash(changed)


def test_fold_dict_round_trips_through_json() -> None:
    closes, volumes = _synthetic_panel()
    result = _run_wfo(closes, volumes)
    payload = [residual_fold_to_dict(f) for f in result.folds]
    assert json.loads(json.dumps(payload, allow_nan=True)) is not None
