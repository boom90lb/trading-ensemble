# src/models/prophet.py
"""Prophet model implementation for time series forecasting."""

import logging
import pickle
import warnings
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
            regressors_to_add = {}
            for feature in X_train.columns:
                # Avoid adding base price/volume columns or target columns as regressors
                if feature not in ["open", "high", "low", "close", "volume"] and not feature.startswith("target_"):
                    try:
                        # Ensure the column exists and can be added
                        if feature in X_train:
                            model.add_regressor(feature)
                            regressors_to_add[feature] = X_train[feature].values
                        else:
                            logger.warning(f"Regressor feature '{feature}' not found in X_train.")
                    except Exception as e:
                        logger.warning(f"Could not add regressor '{feature}': {e}")

            # Add collected regressors to the DataFrame in one go
            if regressors_to_add:
                try:
                    regressor_df = pd.DataFrame(regressors_to_add, index=prophet_df.index)
                    prophet_df = pd.concat([prophet_df, regressor_df], axis=1)
                except ValueError as e:
                    logger.error(f"Error creating/concatenating regressor DataFrame: {e}")
                    # Potentially log shapes for debugging
                    logger.error(
                        f"prophet_df shape: {prophet_df.shape}, "
                        f"regressors_to_add keys: {list(regressors_to_add.keys())}"
                    )
                    # Decide how to handle: raise error, return, or continue without regressors?
                    # For now, let's log and continue without regressors in this specific error case
                    logger.warning("Proceeding with Prophet fit without regressors due to concatenation error.")
                except Exception as e:  # Catch other potential errors
                    logger.error(f"Unexpected error adding regressors: {e}")
                    # Handle appropriately
                    self.is_fitted = False
                    return self

            # Fit the model
            try:
                model.fit(prophet_df)
            except ValueError as e:
                # Check if the error is due to NaNs specifically
                if "Found NaN" in str(e) or "contains NaN" in str(e):
                    logger.error(f"Error fitting Prophet model due to NaNs: {e}")
                    logger.error("NaNs detected in input data for Prophet. Check feature engineering steps.")
                    # Optionally, log which columns have NaNs
                    nan_cols = prophet_df.columns[prophet_df.isna().any()].tolist()
                    if nan_cols:
                        logger.error(f"Columns with NaNs: {nan_cols}")
                else:
                    # Handle other ValueErrors
                    logger.error(f"ValueError fitting Prophet model: {e}")
                self.is_fitted = False
                return self  # Stop fitting if there's a ValueError
            except Exception as e:  # Catch other unexpected errors during fit
                logger.error(f"Unexpected error fitting Prophet model: {e}")
                self.is_fitted = False
                return self  # Stop fitting on other errors

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

            # Prepare regressors if they were used during training
            extra_regressors = getattr(self.model, "extra_regressors", [])
            regressor_data = {}
            if extra_regressors:
                for regressor_name in extra_regressors:
                    if isinstance(regressor_name, str) and regressor_name in X.columns:
                        # Store regressor data, ensuring float type
                        regressor_data[regressor_name] = X[regressor_name].astype(float).values
                    else:
                        logger.warning(f"Skipping invalid or missing regressor: {regressor_name}")

                # Add all prepared regressors to the future dataframe
                if regressor_data:
                    try:
                        # Create DataFrame from collected regressors
                        regressor_df = pd.DataFrame(regressor_data, index=future.index)

                        # Concatenate with the future DataFrame within a context manager to suppress warning
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                            future = pd.concat([future, regressor_df], axis=1)

                    except ValueError as e:
                        logger.error(f"Error creating/concatenating regressor DataFrame for prediction: {e}")
                        # Log shapes for debugging
                        logger.error(
                            f"future shape: {future.shape}, regressor_data keys: {list(regressor_data.keys())}"
                        )
                        logger.warning("Proceeding with Prophet prediction without regressors due to error.")
                        # Ensure 'future' still has the 'ds' column if concatenation fails badly
                        if "ds" not in future.columns:
                            logger.error("Critical error: 'ds' column lost during regressor handling.")
                            return np.array([])  # Cannot proceed without 'ds'
                    except Exception as e:
                        logger.error(f"Unexpected error adding regressors during prediction: {e}")
                        return np.array([])  # Stop prediction on unexpected error

            # Make predictions
            forecast = self.model.predict(future)

            # Return the 'yhat' column (Prophet's predictions)
            # Explicitly convert to numpy array to satisfy type checker
            return np.array(forecast["yhat"].values)  # type: ignore

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
