# src/baselines/tsmom.py
"""Time-series momentum baseline.

LONG if the K-bar return is positive, SHORT if negative, FLAT if exactly zero
or undefined (lookback warmup). Until `prepare(close)` is called the model
returns FLAT.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class TSMOM(BaseModel):
    def __init__(
        self,
        lookback: int = 60,
        target_column: str = "close",
        horizon: int = 1,
        **kwargs: Any,
    ):
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        super().__init__(
            name=f"tsmom_{lookback}",
            target_column=target_column,
            horizon=horizon,
            **kwargs,
        )
        self.lookback = lookback
        self._positions: Optional[pd.Series] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> "TSMOM":
        self.is_fitted = True
        return self

    def prepare(self, close: pd.Series) -> None:
        # pct_change(K) yields NaN for the first K bars — warmup shows up as FLAT.
        ret = close.pct_change(self.lookback)
        sign = np.sign(ret).fillna(0.0).clip(-1.0, 1.0)
        self._positions = sign.astype(float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._positions is None:
            logger.warning("TSMOM.predict called before prepare(); returning FLAT")
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
        out.write_text(f"tsmom lookback={self.lookback}\n")
        return out

    def load(self, model_path: Path) -> "TSMOM":
        self.is_fitted = True
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        return {}
