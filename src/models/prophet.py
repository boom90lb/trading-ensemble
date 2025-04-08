# src/models/prophet.py
"""Prophet model implementation for time series forecasting."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """Prophet model for time series forecasting."""

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 5,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        **kwargs,
    ):
        """Initialize the Prophet model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            daily_seasonality: Whether to include daily seasonality
            weekly_seasonality: Whether to include weekly seasonality
            yearly_seasonality: Whether to include yearly seasonality
            **kwargs: Additional model parameters
        """
        super().__init__(name="prophet", target_column=target_column, horizon=horizon, **kwargs)
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        self.last_date = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "ProphetModel":
        """Fit the Prophet model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()

            # Prophet requires a specific dataframe format with 'ds' and 'y' columns
            prophet_df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})

            # Create and fit the Prophet model
            model = Prophet(
                daily_seasonality="auto",
                weekly_seasonality="auto",
                yearly_seasonality="auto",
            )

            # Add additional regressors if available in X_train
            for feature in X_train.columns:
                if feature not in ["open", "high", "low", "close", "volume"] and not feature.startswith("target_"):
                    model.add_regressor(feature)
                    prophet_df[feature] = X_train[feature].values

            # Fit the model
            model.fit(prophet_df)
            self.model = model
            self.is_fitted = True

            # Store the last date for forecasting
            self.last_date = y_train.index[-1]

            logger.info("Prophet model fitted")

        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted Prophet model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            logger.warning("Prophet model not fitted yet")
            return np.array([])

        try:
            # Create future dataframe for prediction
            future = pd.DataFrame({"ds": X.index})

            # Add regressors if they were used during training
            for regressor in getattr(self.model, "extra_regressors", []):
                regressor_name = regressor["name"]
                if regressor_name in X.columns:
                    future[regressor_name] = X[regressor_name].values

            # Make predictions
            forecast = self.model.predict(future)

            # Return the 'yhat' column (Prophet's predictions)
            return forecast["yhat"].values  # type: ignore

        except Exception as e:
            logger.error(f"Error predicting with Prophet model: {e}")
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        directory = directory / "prophet"
        directory.mkdir(exist_ok=True, parents=True)

        # Create model path
        model_path = directory / f"prophet_h{self.horizon}.pkl"

        # Save the model
        try:
            with open(model_path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "daily_seasonality": self.daily_seasonality,
                        "weekly_seasonality": self.weekly_seasonality,
                        "yearly_seasonality": self.yearly_seasonality,
                        "target_column": self.target_column,
                        "horizon": self.horizon,
                        "is_fitted": self.is_fitted,
                        "last_date": self.last_date,
                        "feature_names": self.feature_names,
                    },
                    f,
                )
            logger.info(f"Prophet model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving Prophet model: {e}")
            return directory

    def load(self, model_path: Path) -> "ProphetModel":
        """Load model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.daily_seasonality = model_data["daily_seasonality"]
            self.weekly_seasonality = model_data["weekly_seasonality"]
            self.yearly_seasonality = model_data["yearly_seasonality"]
            self.target_column = model_data["target_column"]
            self.horizon = model_data["horizon"]
            self.is_fitted = model_data["is_fitted"]
            self.last_date = model_data["last_date"]
            self.feature_names = model_data.get("feature_names", [])

            logger.info(f"Prophet model loaded from {model_path}")
            return self

        except Exception as e:
            logger.error(f"Error loading Prophet model: {e}")
            self.is_fitted = False
            return self
