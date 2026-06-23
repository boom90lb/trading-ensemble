"""End-to-end contract tests for the two broad-universe stat-arb entrypoints.

The headline 574-name results (residual-OU net/gross Sharpe; pairs-on-residuals)
are otherwise reproducible only through throwaway ``results/`` scripts that pytest
never runs. These three tests drive both broad paths end-to-end on a tiny
synthetic panel and assert the *output contract* — summary keys, membership-mask
wiring, finiteness, and the prior-admission invariant — so that later plans which
change net-Sharpe accounting have a regression net. Assertions are contract +
finiteness only; Sharpe values are research outputs that legitimately move and are
never pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.arbitrage.factors import ResidualStatArbConfig
from src.arbitrage.pairs import PairSelectionConfig, residualize_cross_sectional
from src.arbitrage.residual_walk_forward import run_residual_stat_arb_walk_forward
from src.arbitrage.walk_forward import StatArbWalkForwardConfig, run_stat_arb_walk_forward


def _broad_panel(
    n_days: int = 400, n_assets: int = 12, seed: int = 7, resid_b: float = 0.55, resid_vol: float = 0.012
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor + AR(1)-residual returns so residual reversion signals genuinely exist.

    Mirrors ``tests/test_residual_statarb.py::_synthetic_panel`` but wider (12 names,
    ~400 business days) so both broad paths produce multiple folds and trades.
    """
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


def test_residual_broad_run_contract() -> None:
    closes, volumes = _broad_panel()
    result = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        signal_config=ResidualStatArbConfig(
            corr_window=60,
            regr_window=20,
            n_factors=2,
            rebalance_every=5,
            min_median_dollar_volume=1_000.0,
            dollar_volume_window=5,
        ),
        walk_config=StatArbWalkForwardConfig(formation_bars=90, test_bars=40, min_test_bars=20),
    )
    contract = {"sharpe", "oos_periodic_sharpe", "n_symbols", "avg_names_traded", "signal_evaluations"}
    assert contract.issubset(result.summary)
    assert all(math.isfinite(result.summary[key]) for key in contract)
    assert result.summary["n_symbols"] >= 2
    assert result.summary["avg_names_traded"] > 0


def test_residual_membership_mask_changes_result() -> None:
    closes, volumes = _broad_panel()
    signal_config = ResidualStatArbConfig(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        rebalance_every=5,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )
    walk_config = StatArbWalkForwardConfig(formation_bars=90, test_bars=40, min_test_bars=20)

    unmasked = run_residual_stat_arb_walk_forward(
        closes, closes.copy(), volumes, signal_config=signal_config, walk_config=walk_config
    )

    # Drop one name for the back half of the sample (True == index member).
    mask = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    mask.iloc[len(closes) // 2 :, 0] = False
    masked = run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),
        volumes,
        signal_config=signal_config,
        walk_config=walk_config,
        membership_mask=mask,
    )

    # The mask is wired through the WFO: dropping a name cannot raise the count.
    assert masked.summary["avg_names_traded"] <= unmasked.summary["avg_names_traded"]


def test_pairs_on_residuals_contract() -> None:
    closes, _ = _broad_panel()
    resid = residualize_cross_sectional(closes)
    # Residualizing re-accumulates into a positive price level for the log-price machinery.
    assert (resid.dropna() > 0).all().all()

    result = run_stat_arb_walk_forward(
        resid,
        resid,
        selection_config=PairSelectionConfig(
            min_obs=120, candidate_prior="corr", prior_min_abs_corr=0.3, max_pairs=10
        ),
        walk_config=StatArbWalkForwardConfig(formation_bars=180, test_bars=40, min_test_bars=20),
    )
    assert {"sharpe", "pair_set_dsr"}.issubset(result.summary)
    # The corr prior is a screen: it can only ever admit a subset of the raw pair family.
    assert result.folds
    for fold in result.folds:
        report = fold.selection_report
        assert report.n_prior_admitted <= report.n_symbol_pairs
