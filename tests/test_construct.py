"""Unit tests for shared portfolio construction primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.arbitrage.residual import build_book_row
from src.portfolio.construct import (
    build_residual_book_row,
    construct_directional_targets,
    strength_multiplier,
)


def test_residual_book_wrapper_matches_legacy_export() -> None:
    states = np.array([1, -1, 0], dtype=np.int8)
    beta_day = np.array([[0.5, 1.0, 0.0]])
    eigenportfolios = np.array([[0.2, -0.1, 0.3]])

    shared = build_residual_book_row(states, beta_day, eigenportfolios, position_unit=0.02)
    legacy = build_book_row(states, beta_day, eigenportfolios, position_unit=0.02)

    np.testing.assert_allclose(shared, legacy)


def test_strength_multiplier_zero_at_entry_and_caps() -> None:
    out = strength_multiplier(np.array([np.nan, 1.0, 1.5, 4.0]), entry_band=1.0, cap=2.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 2.0])
    with pytest.raises(ValueError):
        strength_multiplier(np.array([1.0]), entry_band=0.0)


def test_directional_construct_does_not_upscale_cash() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="B")
    scores = pd.DataFrame({"AAA": [0.2, 1.0], "BBB": [0.1, -1.0]}, index=idx)

    targets = construct_directional_targets(
        scores,
        position_size=0.5,
        max_gross=1.0,
        max_symbol_abs_weight=0.5,
    )

    assert targets.abs().sum(axis=1).iloc[0] == pytest.approx(0.15)
    assert targets.abs().sum(axis=1).iloc[1] == pytest.approx(1.0)
    assert targets.loc[idx[0], "AAA"] == pytest.approx(0.1)


def test_directional_construct_preserves_missing_decisions() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="B")
    scores = pd.DataFrame({"AAA": [0.2, np.nan]}, index=idx)
    targets = construct_directional_targets(
        scores,
        position_size=0.5,
        max_gross=1.0,
        max_symbol_abs_weight=0.5,
    )
    assert targets.loc[idx[0], "AAA"] == pytest.approx(0.1)
    assert pd.isna(targets.loc[idx[1], "AAA"])
