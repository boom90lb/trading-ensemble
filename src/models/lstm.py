# src/models/lstm.py
"""LSTM model implementation for time series forecasting."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting."""

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 5,
        sequence_length: int = 20,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the LSTM model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            sequence_length: Number of time steps to consider for each prediction
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional model parameters
        """
        super().__init__(name="lstm", target_column=target_column, horizon=horizon, **kwargs)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.feature_names = []

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input.

        Args:
            data: Input data array [n_samples, n_features]

        Returns:
            Tuple of (X, y) where:
            - X is a 3D array [n_sequences, sequence_length, n_features]
            - y is a 1D array [n_sequences]
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])  # Assuming target is the first column

        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        model = Sequential()

        # Add LSTM layers
        model.add(LSTM(self.lstm_units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(self.dropout_rate))

        model.add(LSTM(self.lstm_units // 2, return_sequences=False))
        model.add(Dropout(self.dropout_rate))

        # Add Dense output layer
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")

        return model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "LSTMModel":
        """Fit the LSTM model to training data.

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

            # Combine features and target for sequence creation
            train_data = X_train.copy()
            train_data["target"] = y_train

            # Move target to the first column for sequence creation
            cols = ["target"] + [col for col in train_data.columns if col != "target"]
            train_data = train_data[cols]

            # Convert to numpy array
            train_array = train_data.values

            # Create sequences
            X, y = self._create_sequences(train_array)

            # Handle empty sequences
            if len(X) == 0 or len(y) == 0:
                logger.error("Not enough data to create sequences")
                return self

            # Build the model
            self.model = self._build_model((self.sequence_length, X.shape[2]))

            # Log the shape of the input sequences before fitting
            logger.info(f"Shape of sequences being passed to LSTM fit: {X.shape}")
            logger.info(f"Shape of targets being passed to LSTM fit: {y.shape}")

            # Set up early stopping
            early_stopping = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)

            # Train the model
            self.model.fit(  # type: ignore
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1,
                **kwargs,
            )

            self.is_fitted = True
            logger.info("LSTM model fitted successfully")

        except Exception as e:
            # Log the full traceback for detailed debugging
            logger.error(f"Error fitting LSTM model: {e}", exc_info=True)
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted LSTM model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted or self.model is None:
            logger.warning("LSTM model not fitted yet")
            return np.array([])

        try:
            # Prepare data for prediction
            pred_data = X.copy()

            # If the target column exists, move it to the first column
            if self.target_column in pred_data.columns:
                cols = [self.target_column] + [col for col in pred_data.columns if col != self.target_column]
                pred_data = pred_data[cols]
            else:
                # Add a dummy target column (will be ignored in prediction)
                pred_data.insert(0, "target", 0)

            # Convert to numpy array
            pred_array = pred_data.values

            # Check if we have enough data for a sequence
            if len(pred_array) < self.sequence_length:
                logger.warning(f"Not enough data for prediction (need {self.sequence_length}, got {len(pred_array)})")
                return np.array([])

            # For each position, create a sequence and predict
            predictions = []
            for i in range(len(pred_array) - self.sequence_length + 1):
                seq = pred_array[i : i + self.sequence_length]
                seq = seq.reshape(1, self.sequence_length, seq.shape[1])
                pred = self.model.predict(seq, verbose=0)[0][0]
                predictions.append(pred)

            # If we couldn't make predictions for all positions, pad with zeros
            if len(predictions) < len(X):
                padding = [0] * (len(X) - len(predictions))
                predictions = padding + predictions

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {e}")
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        directory = directory / "lstm"
        directory.mkdir(exist_ok=True, parents=True)

        # Create model path
        model_dir = directory / f"lstm_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)

        # Save the Keras model
        keras_path = model_dir / "keras_model.h5"

        # Save the model parameters
        params_path = model_dir / "params.pkl"

        try:
            if self.is_fitted and self.model is not None:
                # Save Keras model
                self.model.save(keras_path)  # type: ignore

                # Save parameters
                with open(params_path, "wb") as f:
                    pickle.dump(
                        {
                            "target_column": self.target_column,
                            "horizon": self.horizon,
                            "sequence_length": self.sequence_length,
                            "lstm_units": self.lstm_units,
                            "dropout_rate": self.dropout_rate,
                            "learning_rate": self.learning_rate,
                            "is_fitted": self.is_fitted,
                            "feature_names": self.feature_names,
                        },
                        f,
                    )

                logger.info(f"LSTM model saved to {model_dir}")
            else:
                logger.warning("Cannot save model: not fitted yet")

            return model_dir

        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return directory

    def load(self, model_path: Path) -> "LSTMModel":
        """Load model from disk.

        Args:
            model_path: Path to the saved model directory

        Returns:
            Loaded model
        """
        try:
            # Load parameters
            params_path = model_path / "params.pkl"
            with open(params_path, "rb") as f:
                params = pickle.load(f)

            # Update model parameters
            self.target_column = params["target_column"]
            self.horizon = params["horizon"]
            self.sequence_length = params["sequence_length"]
            self.lstm_units = params["lstm_units"]
            self.dropout_rate = params["dropout_rate"]
            self.learning_rate = params["learning_rate"]
            self.is_fitted = params["is_fitted"]
            self.feature_names = params.get("feature_names", [])

            # Load Keras model
            keras_path = model_path / "keras_model.h5"
            self.model = load_model(keras_path)

            logger.info(f"LSTM model loaded from {model_path}")
            return self

        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            self.is_fitted = False
            return self
