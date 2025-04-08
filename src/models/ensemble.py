# src/models/ensemble.py
"""Ensemble model implementation for time series forecasting."""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

from src.config import MODELS_DIR, EnsembleConfig
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple forecasting models."""

    def __init__(self, target_column: str = "close", horizon: int = 5, config: Optional[EnsembleConfig] = None):
        """Initialize the ensemble model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            config: Ensemble configuration
        """
        super().__init__(name="ensemble", target_column=target_column, horizon=horizon)

        # Store configuration
        self.config = config or EnsembleConfig(models=[])

        # Initialize model components
        self.models: Dict[str, BaseModel] = {}  # Dictionary of {model_name: model_instance}
        self.weights: Dict[str, float] = {}  # Dictionary of {model_name: weight}
        self.errors: Dict[str, float] = {}  # Dictionary of {model_name: error_metric}
        self.is_fitted = False

        # Initialize model weights from config
        for model_config in self.config.models:
            if model_config.enabled:
                self.weights[model_config.name] = model_config.weight

        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1."""
        if not self.weights:
            return

        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight

    def _create_model(self, model_name: str) -> Optional[BaseModel]:
        """Create a model instance by name.

        Args:
            model_name: Name of the model to create

        Returns:
            Model instance or None if not supported
        """
        try:
            if model_name == "arima":
                from src.models.arima import ARIMAModel

                return ARIMAModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "prophet":
                from src.models.prophet import ProphetModel

                return ProphetModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "lstm":
                from src.models.lstm import LSTMModel

                return LSTMModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "xgboost":
                from src.models.xgboost_model import XGBoostModel

                return XGBoostModel(target_column=self.target_column, horizon=self.horizon)
            elif model_name == "lstm_ppo":
                # LSTM-PPO is usually imported separately due to RL dependencies
                logger.info("LSTM-PPO model should be imported separately")
                return None
            else:
                logger.warning(f"Unsupported model: {model_name}")
                return None
        except ImportError as e:
            logger.error(f"Could not import {model_name} model: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating {model_name} model: {e}")
            return None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "EnsembleModel":
        """Fit the ensemble model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters for individual models

        Returns:
            Self for method chaining
        """
        # Initialize models if not already present
        for model_name in list(self.weights.keys()):
            if model_name == "lstm_ppo":
                # Skip LSTM-PPO as it should be initialized separately
                continue

            if model_name not in self.models:
                model = self._create_model(model_name)
                if model is not None:
                    self.models[model_name] = model
                else:
                    # Remove model from weights if it couldn't be created
                    logger.warning(f"Removing {model_name} from ensemble due to initialization failure")
                    del self.weights[model_name]

        if not self.models:
            logger.error("No models to fit")
            return self

        # Fit each model
        fitted_models = []
        for name, model in self.models.items():
            # Skip LSTM-PPO as it should be fitted separately
            if name == "lstm_ppo":
                continue

            try:
                logger.info(f"Fitting {name} model")
                # Pass kwargs to individual model fit methods
                model.fit(X_train, y_train, **kwargs)
                fitted_models.append(name)
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}")

        # Update weights based on fitted models
        if self.config.weighting_strategy == "dynamic":
            self._update_weights_dynamic(X_train, y_train, fitted_models)

        # Normalize weights
        self._normalize_weights()

        self.is_fitted = len(fitted_models) > 0
        logger.info(f"Ensemble model fitted with {len(fitted_models)} component models")
        logger.info(f"Model weights: {self.weights}")

        return self

    def _update_weights_dynamic(self, X: pd.DataFrame, y: pd.Series, model_names: List[str]) -> None:
        """Update model weights based on validation performance.

        Args:
            X: Validation features
            y: Validation targets
            model_names: List of fitted model names
        """
        self.errors = {}

        # Calculate errors for each model
        for name in model_names:
            if name not in self.models:
                continue
            model = self.models[name]
            try:
                predictions = model.predict(X)

                # Calculate error metrics
                if len(predictions) > 0 and len(y) == len(predictions):
                    mae = mean_absolute_error(y, predictions)
                    self.errors[name] = float(mae)
                else:
                    logger.warning(f"Prediction length mismatch for {name}. Skipping error calculation.")

            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")

        # Update weights based on inverse error
        if self.errors:
            for name in model_names:
                if name in self.errors and self.errors[name] > 0:
                    # Inverse error weighting
                    self.weights[name] = 1.0 / self.errors[name]  # type: ignore
                else:
                    # If error is zero or not available, use default weight or previous weight
                    # Ensure the key exists before accessing it
                    if name not in self.weights:
                        self.weights[name] = 1.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the ensemble model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted or not self.models:
            logger.warning("Ensemble model not fitted yet")
            return np.array([])

        # Get predictions from each model
        model_predictions = {}
        valid_model_names = []
        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                if len(preds) == len(X):
                    model_predictions[name] = preds
                    valid_model_names.append(name)
                elif len(preds) > 0:
                    logger.warning(
                        f"Prediction length mismatch for {name}. Got {len(preds)}, expected {len(X)}. Skipping."
                    )
                # else: prediction failed, already logged in predict method

            except Exception as e:
                logger.error(f"Error predicting with {name} model: {e}")

        if not model_predictions:
            logger.warning("No valid model predictions available")
            return np.array([])

        # Compute weighted average using only valid predictions
        ensemble_preds = np.zeros(len(X))
        total_weight = 0.0

        for name in valid_model_names:
            if name in self.weights:
                weight = self.weights[name]
                preds = model_predictions[name]
                ensemble_preds += weight * preds
                total_weight += weight

        # Normalize if total weight is not 1 (and > 0)
        if total_weight > 0 and not np.isclose(total_weight, 1.0):
            ensemble_preds /= total_weight
        elif total_weight == 0:
            logger.warning("Total weight of valid models is zero. Returning unweighted average or zeros.")
            # Fallback: simple average of available predictions or return zeros
            if len(valid_model_names) > 0:
                fallback_preds = np.array([model_predictions[name] for name in valid_model_names])
                ensemble_preds = np.mean(fallback_preds, axis=0)
            else:
                ensemble_preds = np.zeros(len(X))  # Should not happen if model_predictions is not empty

        return ensemble_preds

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the ensemble model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            logger.warning("Ensemble model not fitted yet")
            return {}

        try:
            # Get predictions
            y_pred = self.predict(X_test)

            if len(y_pred) == 0 or len(y_pred) != len(y_test):
                logger.warning("Prediction length mismatch or no predictions available for evaluation")
                return {}

            # Calculate metrics
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }

            # Evaluate individual models for comparison
            for name, model in self.models.items():
                try:
                    model_preds = model.predict(X_test)
                    if len(model_preds) == len(y_test):
                        metrics[f"{name}_mae"] = mean_absolute_error(y_test, model_preds)
                        metrics[f"{name}_rmse"] = np.sqrt(mean_squared_error(y_test, model_preds))
                    elif len(model_preds) > 0:
                        logger.warning(f"Individual model evaluation skipped for {name} due to length mismatch.")

                except Exception as e:
                    logger.error(f"Error evaluating {name} model: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            return {}

    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get the contribution of each model to the ensemble prediction.

        Args:
            X: Features

        Returns:
            DataFrame with model contributions
        """
        if not self.is_fitted:
            logger.warning("Ensemble model not fitted yet")
            return pd.DataFrame()

        # Get ensemble prediction
        ensemble_pred = self.predict(X)

        if len(ensemble_pred) != len(X):
            logger.warning("Ensemble prediction length mismatch in get_model_contributions")
            return pd.DataFrame()

        # Get individual model predictions
        model_data = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                if len(preds) == len(X):
                    model_data[f"{name}_pred"] = preds
                    if name in self.weights:
                        weight = self.weights[name]
                        model_data[f"{name}_weight"] = [weight] * len(X)  # type: ignore
                        # Calculate weighted contribution
                        model_data[f"{name}_contrib"] = preds * weight
                elif len(preds) > 0:
                    logger.warning(f"Skipping contributions for {name} due to prediction length mismatch.")

            except Exception as e:
                logger.error(f"Error getting predictions for {name} model in contributions: {e}")

        # Create DataFrame
        if model_data:
            df = pd.DataFrame(model_data, index=X.index)
            df["ensemble_pred"] = ensemble_pred
            return df
        else:
            return pd.DataFrame()

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save the ensemble model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)

        # Create model path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = directory / f"ensemble_h{self.horizon}_{timestamp}.pkl"

        # Create a serializable copy of the model state
        save_state: Dict[str, Any] = {
            "target_column": self.target_column,
            "horizon": self.horizon,
            "config": self.config,
            "weights": self.weights,
            "errors": self.errors,
            "is_fitted": self.is_fitted,
            "models": {},
        }

        # Include fitted models except LSTM-PPO (save separately)
        for name, model_instance in self.models.items():
            if name != "lstm_ppo":
                # Assume models have a get_params method or are pickleable
                # If models are not directly pickleable (e.g., Keras), save their state/path
                save_state["models"][name] = model_instance  # Or model_instance.get_state() / path

        # Save the state
        with open(model_path, "wb") as f:
            pickle.dump(save_state, f)

        logger.info(f"Ensemble model state saved to {model_path}")

        # Note: Individual models (like Keras, XGBoost) might need separate saving
        # if they are not directly pickleable or if saving their state is preferred.
        # LSTM-PPO should definitely be saved separately.

        return model_path

    def load(self, model_path: Path) -> "EnsembleModel":
        """Load model state from disk.

        Args:
            model_path: Path to the saved model state file

        Returns:
            Loaded model instance (self)
        """
        with open(model_path, "rb") as f:
            loaded_state = pickle.load(f)

        # Restore attributes
        self.target_column = loaded_state["target_column"]
        self.horizon = loaded_state["horizon"]
        self.config = loaded_state["config"]
        self.weights = loaded_state["weights"]
        self.errors = loaded_state["errors"]
        self.is_fitted = loaded_state["is_fitted"]
        self.models = loaded_state["models"]  # Assumes models were saved directly or state can be restored

        # Note: If models were saved separately (e.g., Keras path), load them here.
        # Also, need to handle loading LSTM-PPO separately and adding it back.

        logger.info(f"Ensemble model state loaded from {model_path}")

        return self
