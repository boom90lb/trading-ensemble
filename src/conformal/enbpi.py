"""EnbPI-style block-cross-conformal calibrator over OOF positions.

The classical EnbPI of Xu & Xie (2021) bags B models on bootstrap samples
and aggregates leave-one-out residuals. For the trading ensemble that
recipe is impractical (refitting RL members B times is prohibitively
expensive) and the random bootstrap also breaks the time-series block
structure that PurgedWalkForward already enforces.

This module instead implements the natural block-cross-conformal variant
(Foygel-Barber et al. 2021 family) by treating the ensemble's existing
PurgedWalkForward meta-folds as the leave-block-out mechanism. The OOF
position vector produced by ``EnsembleModel._compute_oof_positions`` plus
weighting yields one out-of-sample prediction per training row, computed
from a model that did not see that row. The residuals against the ideal
target are therefore exchangeable conditional on the block structure, and
their empirical quantile gives marginal coverage 1-alpha in the
distribution-free sense.
"""

import numpy as np


class EnbPICalibrator:
    """Two-sided symmetric quantile calibrator over OOF residuals.

    Bands are constructed as ``point ± q_{1-alpha}(|residual|)`` (one-sided
    absolute residual quantile, symmetric). This is the simplest variant
    that respects ``ideal_position``'s natural symmetry around zero.
    """

    def __init__(self) -> None:
        self.abs_residuals: np.ndarray = np.array([], dtype=np.float64)
        self.position_cap: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        oof_predictions: np.ndarray,
        targets: np.ndarray,
        position_cap: float = 1.0,
    ) -> "EnbPICalibrator":
        """Store the empirical distribution of absolute OOF residuals.

        Both arrays must be 1-D, same length, finite. Non-finite rows are
        dropped silently — the meta-learner does the same.
        """
        oof = np.asarray(oof_predictions, dtype=np.float64).ravel()
        tgt = np.asarray(targets, dtype=np.float64).ravel()
        if oof.shape != tgt.shape:
            raise ValueError(
                f"oof_predictions shape {oof.shape} != targets shape {tgt.shape}"
            )
        mask = np.isfinite(oof) & np.isfinite(tgt)
        if mask.sum() < 5:
            raise ValueError(
                f"Need >=5 finite (oof, target) pairs to calibrate; got {mask.sum()}"
            )
        self.abs_residuals = np.abs(tgt[mask] - oof[mask])
        self.position_cap = float(position_cap)
        self.is_fitted = True
        return self

    def quantile(self, alpha: float) -> float:
        """Return q_{1-alpha} of the stored absolute-residual distribution.

        Uses the (1 - alpha) * (n+1) / n finite-sample correction so that
        small calibration sets do not under-cover.
        """
        if not self.is_fitted:
            raise RuntimeError("EnbPICalibrator.quantile called before fit")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        n = len(self.abs_residuals)
        level = min(1.0, (1.0 - alpha) * (n + 1) / n)
        return float(np.quantile(self.abs_residuals, level))

    def band(
        self,
        point: np.ndarray,
        alpha: float,
    ) -> tuple:
        """Return (lower, upper) for a vector of point predictions.

        Bands are clipped to the position cap so the returned interval is
        always a valid subset of [-position_cap, +position_cap].
        """
        q = self.quantile(alpha)
        point_arr = np.asarray(point, dtype=np.float64)
        lower = np.clip(point_arr - q, -self.position_cap, self.position_cap)
        upper = np.clip(point_arr + q, -self.position_cap, self.position_cap)
        return lower, upper
