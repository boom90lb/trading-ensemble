"""Tests for the statistical-arbitrage subsystem."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.arbitrage import (
    PairCandidate,
    PairSelectionConfig,
    PairSignalConfig,
    StatArbWalkForwardConfig,
    backtest_target_weights,
    benjamini_hochberg_mask,
    combine_pair_positions,
    fold_result_to_dict,
    generate_pair_positions,
    run_stat_arb_walk_forward,
    scan_cointegrated_pairs,
)
from src.config import ExecutionConfig


def _synthetic_cointegrated_prices(n: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="America/New_York")
    log_x = np.cumsum(rng.normal(0.0002, 0.01, size=n)) + math.log(100.0)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.82 * spread[i - 1] + rng.normal(0.0, 0.012)
    log_y = 0.15 + 0.92 * log_x + spread
    log_z = np.cumsum(rng.normal(0.0001, 0.02, size=n)) + math.log(40.0)
    return pd.DataFrame({"AAA": np.exp(log_y), "BBB": np.exp(log_x), "ZZZ": np.exp(log_z)}, index=idx)


def _open_from_close(close_prices: pd.DataFrame) -> pd.DataFrame:
    return close_prices.copy()


def _test_only_cointegrated_prices(n: int = 141, formation_bars: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="America/New_York")
    test_n = n - formation_bars
    log_x_test = np.linspace(math.log(50.0), math.log(70.0), test_n)
    spread = 0.01 * np.sin(np.arange(test_n))
    aaa = np.concatenate([np.full(formation_bars, 100.0), np.exp(0.25 + log_x_test + spread)])
    bbb = np.concatenate([np.full(formation_bars, 50.0), np.exp(log_x_test)])
    return pd.DataFrame({"AAA": aaa, "BBB": bbb}, index=idx)


def _formation_pair_broken_test_prices(n: int = 161, formation_bars: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="America/New_York")
    log_x = np.cumsum(rng.normal(0.0002, 0.008, size=n)) + math.log(80.0)
    spread = np.zeros(formation_bars)
    for i in range(1, formation_bars):
        spread[i] = 0.65 * spread[i - 1] + rng.normal(0.0, 0.006)
    log_y = np.empty(n)
    log_y[:formation_bars] = 0.10 + 0.95 * log_x[:formation_bars] + spread
    drift = np.linspace(0.0, 0.8, n - formation_bars)
    log_y[formation_bars:] = log_y[formation_bars - 1] + drift
    return pd.DataFrame({"AAA": np.exp(log_y), "BBB": np.exp(log_x)}, index=idx)


def test_benjamini_hochberg_mask_accepts_prefix_under_cutoff() -> None:
    mask = benjamini_hochberg_mask([0.001, 0.03, 0.20, 0.80], alpha=0.05)
    assert mask.tolist() == [True, False, False, False]


def test_scan_cointegrated_pairs_uses_stationarity_filters() -> None:
    prices = _synthetic_cointegrated_prices()
    cfg = PairSelectionConfig(
        min_obs=500,
        max_pairs=3,
        fdr_alpha=0.20,
        min_abs_return_corr=0.2,
        max_half_life=80.0,
        max_beta_drift=0.75,
    )
    pairs = scan_cointegrated_pairs(prices, cfg)
    assert pairs
    assert {pairs[0].asset_y, pairs[0].asset_x} == {"AAA", "BBB"}
    assert pairs[0].coint_pvalue < 0.05
    assert pairs[0].adf_pvalue < 0.05
    assert 1.0 <= pairs[0].half_life <= 80.0


def test_pair_positions_are_hedged_and_stateful() -> None:
    idx = pd.date_range("2026-01-01", periods=8, freq="B", tz="America/New_York")
    candidate = PairCandidate(
        asset_y="AAA",
        asset_x="BBB",
        alpha=0.0,
        beta=1.0,
        coint_pvalue=0.01,
        coint_stat=-5.0,
        adf_pvalue=0.01,
        adf_stat=-5.0,
        half_life=5.0,
        beta_drift=0.1,
        return_corr=0.9,
        n_obs=8,
    )
    prices = pd.DataFrame(
        {
            "AAA": np.exp([0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6, 0.0]),
            "BBB": np.ones(8),
        },
        index=idx,
    )
    out = generate_pair_positions(
        prices,
        candidate,
        PairSignalConfig(z_window=3, min_z_obs=3, entry_z=1.0, exit_z=0.2, stop_z=3.0),
    )
    nonzero = out.target_weights[out.target_weights.abs().sum(axis=1) > 0]
    assert not nonzero.empty
    assert np.allclose(nonzero.abs().sum(axis=1), 1.0)
    assert np.allclose(nonzero.sum(axis=1), 0.0)


def test_combine_pair_positions_caps_symbol_and_gross() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    one = pd.DataFrame({"AAA": [0.8, 0.8, 0.0], "BBB": [-0.8, -0.8, 0.0]}, index=idx)
    two = pd.DataFrame({"AAA": [0.8, 0.8, 0.0], "CCC": [-0.8, -0.8, 0.0]}, index=idx)
    combined = combine_pair_positions([one, two], max_gross=1.0, max_symbol_abs_weight=0.4)
    assert combined.abs().sum(axis=1).max() <= 1.0 + 1e-12
    assert combined.abs().max().max() <= 0.4 + 1e-12


def test_backtest_target_weights_no_same_bar_pnl() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="B", tz="America/New_York")
    opens = pd.DataFrame({"AAA": [100.0, 200.0, 200.0]}, index=idx)
    targets = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=idx)
    result = backtest_target_weights(
        opens, targets, execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0)
    )
    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] == 0.0


def test_backtest_target_weights_earns_after_next_open_fill() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="B", tz="America/New_York")
    opens = pd.DataFrame({"AAA": [100.0, 100.0, 110.0]}, index=idx)
    targets = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=idx)
    result = backtest_target_weights(
        opens, targets, execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0)
    )
    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] == pytest.approx(0.10)
    assert result.metrics["total_return"] == pytest.approx(0.10)


def test_backtest_costs_reduce_returns_on_turnover() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="B", tz="America/New_York")
    opens = pd.DataFrame({"AAA": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    targets = pd.DataFrame({"AAA": [1.0, -1.0, 0.0, 0.0]}, index=idx)
    no_cost = backtest_target_weights(
        opens,
        targets,
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )
    costly = backtest_target_weights(
        opens,
        targets,
        execution=ExecutionConfig(spread_bps=5, commission_bps=2, slippage_coeff=10, borrow_rate_bps_annual=100),
    )
    assert no_cost.returns.sum() == 0.0
    assert costly.returns.sum() < 0.0
    assert costly.costs["total"].sum() > 0.0


def test_walk_forward_does_not_select_pair_that_exists_only_in_test() -> None:
    prices = _test_only_cointegrated_prices()
    result = run_stat_arb_walk_forward(
        prices,
        _open_from_close(prices),
        selection_config=PairSelectionConfig(
            min_obs=80,
            max_pairs=2,
            fdr_alpha=0.20,
            min_abs_return_corr=0.0,
            max_beta_drift=10.0,
            max_half_life=1_000.0,
        ),
        signal_config=PairSignalConfig(z_window=10, min_z_obs=5, entry_z=0.5, exit_z=0.1, stop_z=5.0),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=60, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )

    assert len(result.folds) == 1
    assert result.folds[0].selected_pairs == ()
    assert result.folds[0].selection_report.n_symbol_pairs == 1
    assert result.summary["raw_pair_candidates"] == 0.0


def test_walk_forward_reports_broken_test_pair_without_reselecting() -> None:
    prices = _formation_pair_broken_test_prices()
    result = run_stat_arb_walk_forward(
        prices,
        _open_from_close(prices),
        selection_config=PairSelectionConfig(
            min_obs=100,
            max_pairs=1,
            fdr_alpha=0.50,
            min_abs_return_corr=0.0,
            max_beta_drift=1.0,
            max_half_life=1_000.0,
        ),
        signal_config=PairSignalConfig(z_window=20, min_z_obs=10, entry_z=0.5, exit_z=0.1, stop_z=10.0),
        walk_config=StatArbWalkForwardConfig(formation_bars=100, test_bars=60, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )

    fold = result.folds[0]
    assert len(fold.selected_pairs) == 1
    assert fold.selection_report.n_raw_candidates == 1
    assert len(fold.test_pair_diagnostics) == 1
    assert fold.test_pair_diagnostics[0]["spread_std"] is not None
    assert fold.metrics["total_return"] < 0.0


def test_walk_forward_flattens_fold_boundary_targets() -> None:
    prices = _synthetic_cointegrated_prices(141)
    result = run_stat_arb_walk_forward(
        prices[["AAA", "BBB"]],
        _open_from_close(prices[["AAA", "BBB"]]),
        selection_config=PairSelectionConfig(
            min_obs=80,
            max_pairs=1,
            fdr_alpha=0.50,
            min_abs_return_corr=0.0,
            max_beta_drift=1.0,
            max_half_life=1_000.0,
        ),
        signal_config=PairSignalConfig(z_window=12, min_z_obs=6, entry_z=0.5, exit_z=0.1, stop_z=10.0),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=30, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )

    assert len(result.folds) == 2
    assert result.folds[0].selected_pairs
    first_fold_end = result.folds[0].test_end
    second_fold_start = result.folds[1].test_start
    assert result.portfolio.target_weights.loc[first_fold_end].abs().sum() == 0.0
    assert result.portfolio.fill_weights.loc[second_fold_start].abs().sum() == 0.0


def test_walk_forward_selection_metadata_is_deterministic() -> None:
    prices = _test_only_cointegrated_prices()
    kwargs = dict(
        selection_config=PairSelectionConfig(
            min_obs=80,
            max_pairs=2,
            fdr_alpha=0.20,
            min_abs_return_corr=0.0,
            max_beta_drift=10.0,
            max_half_life=1_000.0,
        ),
        signal_config=PairSignalConfig(z_window=10, min_z_obs=5, entry_z=0.5, exit_z=0.1, stop_z=5.0),
        walk_config=StatArbWalkForwardConfig(formation_bars=80, test_bars=60, min_test_bars=20),
        execution=ExecutionConfig(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0),
    )
    first = run_stat_arb_walk_forward(prices, _open_from_close(prices), **kwargs)
    second = run_stat_arb_walk_forward(prices, _open_from_close(prices), **kwargs)

    first_reports = [fold_result_to_dict(fold)["selection_report"] for fold in first.folds]
    second_reports = [fold_result_to_dict(fold)["selection_report"] for fold in second.folds]
    assert first_reports == second_reports
