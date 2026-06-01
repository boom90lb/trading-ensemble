"""Purged + embargoed walk-forward cross-validation for time-series.

Implements López de Prado (AFML §7.4) walk-forward with two distinct
leakage controls:

  * **purge_horizon** — drop training samples whose label window overlaps
    the test fold. With a forward-return target `y_t = (p_{t+h} - p_t)/p_t`,
    every training label for `t ∈ [test_start - h, test_start - 1]` peeks
    into the test region; those rows must be removed from train.

  * **embargo_pct** — additional buffer (as fraction of total bars) appended
    after each test fold and excluded from subsequent training. Targets
    serial correlation in model residuals across adjacent folds, not label
    overlap (purge already handles that).

The splitter yields integer-position index arrays compatible with
`pd.DataFrame.iloc`. It does not assume the input has a monotonic
DatetimeIndex (the embargo is in bar count, not wall-clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sized, Tuple

import numpy as np


@dataclass
class PurgedWalkForward:
    """Walk-forward splitter with purge and embargo.

    Parameters
    ----------
    n_splits : int
        Number of test folds. The series is partitioned into ``n_splits + 1``
        equal chunks; chunk 0 is reserved as the initial training pool and
        chunks ``1..n_splits`` each become one test fold.
    purge_horizon : int
        Number of bars whose forward-looking labels overlap a test fold.
        Set to the label horizon `h` (e.g. 5 for a 5-day forward return).
        Training rows whose label window touches the test region are dropped.
    embargo_pct : float, default 0.01
        Fraction of total bars to skip *after* each test fold. The embargoed
        region is excluded from any subsequent fold's training set. Set to
        ``0.0`` to disable.
    expanding : bool, default True
        If True, each fold's train set starts at index 0 (expanding window).
        If False, train is a rolling window of length ``test_size * 2``
        ending immediately before the purge zone.
    """

    n_splits: int
    purge_horizon: int
    embargo_pct: float = 0.01
    expanding: bool = True

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits}")
        if self.purge_horizon < 0:
            raise ValueError(
                f"purge_horizon must be >= 0, got {self.purge_horizon}"
            )
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError(
                f"embargo_pct must be in [0, 1), got {self.embargo_pct}"
            )

    def get_n_splits(self, X: Sized | None = None) -> int:
        return self.n_splits

    def split(
        self, X: Sized
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer arrays for each fold."""
        n = len(X)
        if n < (self.n_splits + 1) * 2:
            raise ValueError(
                f"Need at least {(self.n_splits + 1) * 2} samples for "
                f"n_splits={self.n_splits}; got {n}"
            )
        embargo = int(self.embargo_pct * n)
        test_size = n // (self.n_splits + 1)

        prior_embargo_zones: list[tuple[int, int]] = []

        for k in range(self.n_splits):
            test_start = (k + 1) * test_size
            if k < self.n_splits - 1:
                test_end = test_start + test_size
            else:
                test_end = n
            train_end = test_start - self.purge_horizon
            if train_end <= 0:
                prior_embargo_zones.append(
                    (test_end, min(test_end + embargo, n))
                )
                continue

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - test_size * 2)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train_start:train_end] = True

            for emb_start, emb_end in prior_embargo_zones:
                train_mask[emb_start:emb_end] = False

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0:
                prior_embargo_zones.append(
                    (test_end, min(test_end + embargo, n))
                )
                continue

            yield train_idx, test_idx
            prior_embargo_zones.append(
                (test_end, min(test_end + embargo, n))
            )
