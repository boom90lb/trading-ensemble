# src/models/arima.py
"""ARIMA model implementation for time series forecasting."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 5,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        use_exog: bool = False,
        auto_arima: bool = False,
        **kwargs,
    ):
        """Initialize the ARIMA model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            use_exog: Whether to use exogenous variables
            auto_arima: Whether to automatically determine ARIMA order
            **kwargs: Additional model parameters
        """
        super().__init__(name="arima", target_column=target_column, horizon=horizon, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.auto_arima = auto_arima
        self.model = None
        self.last_observations: Optional[np.ndarray] = None
        self.last_date = None
        self.exog_columns: List[str] = []

    def _select_exog_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Select exogenous features for ARIMA.

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with selected exogenous features or None
        """
        # Exclude target, price, and lagged features
        excluded_patterns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target_",
            "direction_",
            "ma",
            "ema",  # Moving averages often cause multicollinearity
        ]

        exog_columns = []
        for col in X.columns:
            if not any(pattern in col.lower() for pattern in excluded_patterns):
                exog_columns.append(col)

        self.exog_columns = exog_columns

        if exog_columns:
            return X[exog_columns]
        else:
            return None

    def _determine_best_order(
        self, y: pd.Series, exog: Optional[pd.DataFrame] = None
    ) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """Determine the best ARIMA order using information criteria.

        Args:
            y: Target series
            exog: Exogenous variables

        Returns:
            Tuple of (order, seasonal_order)
        """
        from itertools import product

        # Define candidate orders
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None

        # Try different orders
        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(y, order=(p, d, q), exog=exog)
                results = model.fit()
                aic = results.aic  # type: ignore

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except Exception as e:
                continue

        # If data is seasonal, try seasonal orders
        if len(y) >= 365:  # Need enough data for seasonality
            seasonal_periods = [7, 12, 52]  # Weekly, monthly, yearly

            for s in seasonal_periods:
                if len(y) >= 2 * s:  # Need at least 2 full seasonal cycles
                    for P, D, Q in product(range(0, 2), range(0, 2), range(0, 2)):
                        try:
                            model = SARIMAX(y, order=best_order, seasonal_order=(P, D, Q, s), exog=exog)
                            results = model.fit(disp=False)
                            aic = results.aic  # type: ignore

                            if aic < best_aic:
                                best_aic = aic
                                best_seasonal_order = (P, D, Q, s)
                        except Exception as e:
                            continue

        if best_order is None:
            best_order = self.order  # Fall back to default

        return best_order, best_seasonal_order

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "ARIMAModel":
        """Fit the ARIMA model to training data.

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

            # Use y_train as the target time series
            if y_train is None or len(y_train) == 0:
                logger.error("No training data provided")
                return self

            # Get exogenous variables if requested
            exog = None
            if self.use_exog:
                exog_df = self._select_exog_features(X_train)
                if exog_df is not None and not exog_df.empty:
                    exog = exog_df

            # Determine best order if auto_arima is True
            if self.auto_arima:
                self.order, self.seasonal_order = self._determine_best_order(y_train, exog)
                logger.info(f"Selected order={self.order}, seasonal_order={self.seasonal_order}")

            # Create and fit the ARIMA model
            if self.seasonal_order is not None:
                if self.use_exog and exog is not None:
                    model = SARIMAX(y_train, order=self.order, seasonal_order=self.seasonal_order, exog=exog)
                else:
                    model = SARIMAX(y_train, order=self.order, seasonal_order=self.seasonal_order)
            else:
                if self.use_exog and exog is not None:
                    model = ARIMA(y_train, order=self.order, exog=exog)
                else:
                    model = ARIMA(y_train, order=self.order)

            self.model = model.fit()
            self.is_fitted = True

            # Store the last observations for forecasting
            self.last_observations = y_train.values  # type: ignore
            self.last_date = y_train.index[-1]

            logger.info(f"ARIMA model fitted with order={self.order}, seasonal_order={self.seasonal_order}")

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted ARIMA model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            logger.warning("ARIMA model not fitted yet")
            return np.array([])

        try:
            # For ARIMA with exogenous variables
            exog = None
            if self.use_exog and self.exog_columns:
                # Get exogenous variables for prediction
                exog_cols = [col for col in self.exog_columns if col in X.columns]
                if exog_cols:
                    exog = X[exog_cols]

            # Make forecast
            if hasattr(self.model, "forecast"):
                if exog is not None:
                    forecast = self.model.forecast(steps=len(X), exog=exog)  # type: ignore
                else:
                    forecast = self.model.forecast(steps=len(X))  # type: ignore

                return forecast.values
            else:
                # For older statsmodels versions
                if exog is not None:
                    forecast = self.model.get_forecast(steps=len(X), exog=exog)  # type: ignore
                else:
                    forecast = self.model.get_forecast(steps=len(X))  # type: ignore

                return forecast.predicted_mean.values

        except Exception as e:
            logger.error(f"Error predicting with ARIMA model: {e}")
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        directory = directory / "arima"
        directory.mkdir(exist_ok=True, parents=True)

        # Create model path
        model_path = directory / f"arima_h{self.horizon}.pkl"

        # Save the model
        try:
            with open(model_path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "order": self.order,
                        "seasonal_order": self.seasonal_order,
                        "use_exog": self.use_exog,
                        "auto_arima": self.auto_arima,
                        "target_column": self.target_column,
                        "horizon": self.horizon,
                        "is_fitted": self.is_fitted,
                        "last_observations": self.last_observations,
                        "last_date": self.last_date,
                        "feature_names": self.feature_names,
                        "exog_columns": self.exog_columns,
                    },
                    f,
                )
            logger.info(f"ARIMA model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving ARIMA model: {e}")
            return directory

    def load(self, model_path: Path) -> "ARIMAModel":
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
            self.order = model_data["order"]
            self.seasonal_order = model_data["seasonal_order"]
            self.use_exog = model_data.get("use_exog", False)
            self.auto_arima = model_data.get("auto_arima", False)
            self.target_column = model_data["target_column"]
            self.horizon = model_data["horizon"]
            self.is_fitted = model_data["is_fitted"]
            self.last_observations = model_data["last_observations"]
            self.last_date = model_data["last_date"]
            self.feature_names = model_data.get("feature_names", [])
            self.exog_columns = model_data.get("exog_columns", [])

            logger.info(f"ARIMA model loaded from {model_path}")
            return self

        except Exception as e:
            logger.error(f"Error loading ARIMA model: {e}")
            self.is_fitted = False
            return self
