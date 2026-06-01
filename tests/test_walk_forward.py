"""Property tests for PurgedWalkForward.

Verifies the two leakage controls actually do what they claim:
purge keeps test-overlapping training labels out of train, and embargo
keeps a buffer of bars after each test out of subsequent train sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import PurgedWalkForward


@pytest.fixture
def series_1000() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(1000)})


def test_n_splits_yielded(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.0)
    folds = list(wfo.split(series_1000))
    assert len(folds) == 5


def test_purge_horizon_enforced(series_1000: pd.DataFrame) -> None:
    """No training row may sit within `purge_horizon` bars before test_start."""
    h = 10
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=h, embargo_pct=0.0)
    for train_idx, test_idx in wfo.split(series_1000):
        test_start = int(test_idx.min())
        assert int(train_idx.max()) + h <= test_start, (
            f"Purge violated: train_max={train_idx.max()} + h={h} > test_start={test_start}"
        )


def test_train_and_test_disjoint(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.01)
    for train_idx, test_idx in wfo.split(series_1000):
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_test_folds_non_overlapping(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.0)
    seen: set[int] = set()
    for _, test_idx in wfo.split(series_1000):
        as_set = set(int(i) for i in test_idx)
        assert seen.isdisjoint(as_set), "Test folds overlap"
        seen.update(as_set)


def test_embargo_excludes_prior_test_zones(series_1000: pd.DataFrame) -> None:
    """Embargo after fold k's test must be excluded from every later fold's
    train. (Effect appears at fold k+2+ because fold k+1's train naturally
    ends at fold k+1's test_start, which is where fold k's embargo begins
    when test slices are adjacent.)
    """
    n = 1000
    embargo_pct = 0.02
    embargo = int(embargo_pct * n)
    wfo = PurgedWalkForward(
        n_splits=5, purge_horizon=5, embargo_pct=embargo_pct
    )
    folds = list(wfo.split(series_1000))
    for i, (_, test_i) in enumerate(folds):
        embargo_zone = set(range(
            int(test_i.max()) + 1,
            int(test_i.max()) + 1 + embargo,
        ))
        for later_train, _ in folds[i + 2:]:
            assert embargo_zone.isdisjoint(
                set(int(j) for j in later_train)
            ), f"Embargo zone from fold {i} leaked into a later fold's train"


def test_expanding_window_grows(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=4, purge_horizon=5, embargo_pct=0.0, expanding=True)
    train_sizes = [len(tr) for tr, _ in wfo.split(series_1000)]
    assert train_sizes == sorted(train_sizes), f"Expanding train should grow: {train_sizes}"


def test_rolling_window_bounded(series_1000: pd.DataFrame) -> None:
    """Rolling window train length is bounded by 2 × test_size (less purge)."""
    wfo = PurgedWalkForward(n_splits=4, purge_horizon=5, embargo_pct=0.0, expanding=False)
    test_size = len(series_1000) // (wfo.n_splits + 1)
    for train_idx, _ in wfo.split(series_1000):
        assert len(train_idx) <= test_size * 2


def test_rejects_bad_args(series_1000: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedWalkForward(n_splits=0, purge_horizon=5)
    with pytest.raises(ValueError, match="purge_horizon"):
        PurgedWalkForward(n_splits=5, purge_horizon=-1)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=1.5)


def test_rejects_too_short_input() -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5)
    with pytest.raises(ValueError, match="at least"):
        list(wfo.split(pd.DataFrame({"x": np.arange(8)})))


def test_zero_embargo_is_noop(series_1000: pd.DataFrame) -> None:
    """embargo=0 → fold k+2's train includes the bar right after fold k's
    test (which would otherwise be embargoed).
    """
    wfo = PurgedWalkForward(n_splits=3, purge_horizon=0, embargo_pct=0.0)
    folds = list(wfo.split(series_1000))
    _, test_0 = folds[0]
    later_train, _ = folds[2]
    assert int(test_0.max()) + 1 in set(int(i) for i in later_train)


def test_embargo_actually_excludes(series_1000: pd.DataFrame) -> None:
    """Positive control for the noop test: embargo>0 → fold k+2's train
    excludes the bar right after fold k's test.
    """
    wfo = PurgedWalkForward(
        n_splits=3, purge_horizon=0, embargo_pct=0.02
    )
    folds = list(wfo.split(series_1000))
    _, test_0 = folds[0]
    later_train, _ = folds[2]
    assert int(test_0.max()) + 1 not in set(int(i) for i in later_train)
