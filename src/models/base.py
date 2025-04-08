# src/models/base.py
"""Base model class for time series forecasting models."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for time series forecasting models."""

    def __init__(self, name: str, target_column: str = "close", horizon: int = 5, **kwargs):
        """Initialize the base model.

        Args:
            name: Model name
            target_column: Target column to predict
            horizon: Forecast horizon
            **kwargs: Additional model parameters
        """
        self.name = name
        self.target_column = target_column
        self.horizon = horizon
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "BaseModel":
        """Fit the model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        raise NotImplementedError("Subclasses must implement save method")

    @abstractmethod
    def load(self, model_path: Path) -> "BaseModel":
        """Load model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement load method")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

        # Get predictions
        y_pred = self.predict(X_test)

        if len(y_pred) == 0:
            logger.warning("No predictions available for evaluation")
            return {}

        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "name": self.name,
            "target_column": self.target_column,
            "horizon": self.horizon,
            "is_fitted": self.is_fitted,
        }

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter: {key}")

        return self
