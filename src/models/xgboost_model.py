# src/models/xgboost_model.py
"""XGBoost model implementation for time series forecasting."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting."""

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        **kwargs,
    ):
        """Initialize the XGBoost model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction required to make a further partition
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            **kwargs: Additional model parameters
        """
        super().__init__(name="xgboost", target_column=target_column, horizon=horizon, **kwargs)

        # XGBoost parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # Model
        self.model: Optional[xgb.Booster] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "XGBoostModel":
        """Fit the XGBoost model to training data.

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

            # Handle missing values
            X_train_clean = X_train.fillna(0)
            y_train_clean = y_train.fillna(y_train.mean())

            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_clean, label=y_train_clean)

            # Set up parameters
            params = {
                "objective": "reg:squarederror",
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "gamma": self.gamma,
                "alpha": self.reg_alpha,
                "lambda": self.reg_lambda,
                "tree_method": "hist",  # Faster training for large datasets
            }

            # Train the model
            self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators, **kwargs)

            self.is_fitted = True
            logger.info("XGBoost model fitted")

        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {e}")
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted XGBoost model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            logger.warning("XGBoost model not fitted yet")
            return np.array([])

        try:
            # Handle missing values
            X_clean = X.fillna(0)

            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X_clean)

            # Make predictions
            predictions = self.model.predict(dtest)

            return predictions

        except Exception as e:
            logger.error(f"Error predicting with XGBoost model: {e}")
            return np.array([])

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model.

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted or self.model is None:
            logger.warning("XGBoost model not fitted yet")
            return pd.DataFrame()

        try:
            # Get feature importance
            importance = self.model.get_score(importance_type="gain")

            # Create DataFrame
            importance_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})

            # Sort by importance
            importance_df = importance_df.sort_values("importance", ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        directory = directory / "xgboost"
        directory.mkdir(exist_ok=True, parents=True)

        # Create model path
        model_path = directory / f"xgboost_h{self.horizon}"
        model_path.mkdir(exist_ok=True, parents=True)

        # Save the model and parameters
        try:
            if self.is_fitted and self.model is not None:
                # Save XGBoost model
                self.model.save_model(str(model_path / "model.json"))

                # Save parameters
                params = {
                    "target_column": self.target_column,
                    "horizon": self.horizon,
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "min_child_weight": self.min_child_weight,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "gamma": self.gamma,
                    "reg_alpha": self.reg_alpha,
                    "reg_lambda": self.reg_lambda,
                    "is_fitted": self.is_fitted,
                    "feature_names": self.feature_names,
                }

                with open(model_path / "params.pkl", "wb") as f:
                    pickle.dump(params, f)

                logger.info(f"XGBoost model saved to {model_path}")
            else:
                logger.warning("Cannot save model: not fitted yet")

            return model_path

        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
            return directory

    def load(self, model_path: Path) -> "XGBoostModel":
        """Load model from disk.

        Args:
            model_path: Path to the saved model directory

        Returns:
            Loaded model
        """
        try:
            # Load parameters
            with open(model_path / "params.pkl", "rb") as f:
                params = pickle.load(f)

            # Update model parameters
            self.target_column = params["target_column"]
            self.horizon = params["horizon"]
            self.n_estimators = params["n_estimators"]
            self.learning_rate = params["learning_rate"]
            self.max_depth = params["max_depth"]
            self.min_child_weight = params["min_child_weight"]
            self.subsample = params["subsample"]
            self.colsample_bytree = params["colsample_bytree"]
            self.gamma = params["gamma"]
            self.reg_alpha = params["reg_alpha"]
            self.reg_lambda = params["reg_lambda"]
            self.is_fitted = params["is_fitted"]
            self.feature_names = params.get("feature_names", [])

            # Load XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(str(model_path / "model.json"))  # type: ignore

            logger.info(f"XGBoost model loaded from {model_path}")
            return self

        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            self.is_fitted = False
            return self
