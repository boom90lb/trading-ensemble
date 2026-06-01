# src/baselines/buy_and_hold.py
"""Buy-and-hold baseline: constant LONG (+1) position."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class BuyAndHold(BaseModel):
    """Always LONG. The minimum benchmark every backtest must beat."""

    def __init__(self, target_column: str = "close", horizon: int = 1, **kwargs: Any):
        super().__init__(name="buy_and_hold", target_column=target_column, horizon=horizon, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> "BuyAndHold":
        self.is_fitted = True
        return self

    def prepare(self, close: pd.Series) -> None:
        del close

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=float)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"{self.name}.txt"
        out.write_text("buy_and_hold\n")
        return out

    def load(self, model_path: Path) -> "BuyAndHold":
        self.is_fitted = True
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        return {}
