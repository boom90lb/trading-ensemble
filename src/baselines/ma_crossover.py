# src/baselines/ma_crossover.py
"""Moving-average crossover baseline.

LONG when fast SMA > slow SMA, SHORT otherwise. Until `prepare(close)` is
called the model returns FLAT.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MACrossover(BaseModel):
    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        target_column: str = "close",
        horizon: int = 1,
        **kwargs: Any,
    ):
        if fast < 1 or slow < 1:
            raise ValueError(f"fast and slow must be >= 1, got fast={fast}, slow={slow}")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        super().__init__(
            name=f"ma_crossover_{fast}_{slow}",
            target_column=target_column,
            horizon=horizon,
            **kwargs,
        )
        self.fast = fast
        self.slow = slow
        self._positions: Optional[pd.Series] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> "MACrossover":
        self.is_fitted = True
        return self

    def prepare(self, close: pd.Series) -> None:
        # min_periods=slow ensures NaN until the slow window is filled — the lookback
        # warmup shows up as FLAT positions at the start of the backtest, not a leak.
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
        sign = np.sign(fast_ma - slow_ma).fillna(0.0).clip(-1.0, 1.0)
        self._positions = sign.astype(float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._positions is None:
            logger.warning("MACrossover.predict called before prepare(); returning FLAT")
            return np.zeros(len(X), dtype=float)
        out = np.zeros(len(X), dtype=float)
        for i, idx in enumerate(X.index):
            if idx in self._positions.index:
                out[i] = float(self._positions.loc[idx])
        return out

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"{self.name}.txt"
        out.write_text(f"ma_crossover fast={self.fast} slow={self.slow}\n")
        return out

    def load(self, model_path: Path) -> "MACrossover":
        self.is_fitted = True
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        return {}
