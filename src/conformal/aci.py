"""Adaptive Conformal Inference (Gibbs & Candes 2021) online quantile adapter.

ACI maintains a running ``alpha_t`` that drifts upward (wider bands) after
miscoverage events and downward (tighter bands) when realized outcomes
keep landing inside the band. Coverage guarantees hold asymptotically
even under arbitrary distribution shift -- the key property EnbPI alone
lacks.

Update rule:
    alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{miss_t})

clamped to a small open interval inside (0, 1) so the quantile call on
the EnbPI calibrator never degenerates.
"""

from dataclasses import dataclass


_ALPHA_FLOOR = 1e-4
_ALPHA_CEIL = 1.0 - 1e-4


@dataclass
class ACIState:
    """In-place state for one online conformal quantile track.

    Attributes:
        alpha_target: The target miscoverage rate (e.g. 0.10 for 90% bands).
        gamma: ACI learning rate. Default 0.01 follows Gibbs & Candes.
        alpha_t: The running quantile parameter; reads back as ``current_alpha()``.
    """

    alpha_target: float = 0.10
    gamma: float = 0.01
    alpha_t: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_target < 1.0:
            raise ValueError(
                f"alpha_target must be in (0, 1); got {self.alpha_target}"
            )
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0; got {self.gamma}")
        # Initialize alpha_t to the target if unset (caller may override).
        if not 0.0 < self.alpha_t < 1.0:
            self.alpha_t = self.alpha_target

    def update(self, in_band: bool) -> float:
        """Apply one ACI step. Returns the new alpha_t.

        Convention: bands are calibrated at the (1 - alpha_t) quantile.
        Smaller alpha_t -> larger quantile -> WIDER band.

        in_band=True  -> miss=0 -> alpha_t GROWS by gamma*alpha_target  -> tighter bands next time.
        in_band=False -> miss=1 -> alpha_t SHRINKS by gamma*(1 - alpha_target) -> wider bands next time.
        """
        miss = 0.0 if in_band else 1.0
        new_alpha = self.alpha_t + self.gamma * (self.alpha_target - miss)
        self.alpha_t = float(min(_ALPHA_CEIL, max(_ALPHA_FLOOR, new_alpha)))
        return self.alpha_t

    def current_alpha(self) -> float:
        return self.alpha_t
