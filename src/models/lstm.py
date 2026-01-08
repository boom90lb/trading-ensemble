# src/models/lstm.py
"""LSTM model implementation using Flax/JAX for time series forecasting."""

import logging
import pickle

# Import necessary modules
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pandas as pd

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LSTMModule(nn.Module):
    """LSTM neural network module using Flax linen."""

    input_features: int = 1
    lstm_units: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, training: bool = False):
        """Forward pass through the LSTM module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, features]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size]
        """
        # First LSTM layer
        lstm1 = nn.RNN(
            cell=nn.LSTMCell(features=self.lstm_units),
            return_carry=False,
        )
        outputs1 = lstm1(x)

        # Apply dropout after first LSTM layer
        outputs1 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(outputs1)

        # Second LSTM layer
        lstm2 = nn.RNN(
            cell=nn.LSTMCell(features=self.lstm_units // 2),
            return_carry=False,
        )
        outputs2 = lstm2(outputs1)

        # Apply dropout after second LSTM layer
        outputs2 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(outputs2)

        # Get the last output from the sequence
        last_output = outputs2[:, -1, :]

        # Apply final dense layer
        y = nn.Dense(features=1)(last_output)
        return y.squeeze(-1)


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting using JAX/Flax NNX."""

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
        seed: int = 42,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the LSTM model."""
        super().__init__(name="lstm", target_column=target_column, horizon=horizon, **kwargs)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        # Initialize seed for reproducibility
        self.rng = jax.random.key(seed)
        self.model = None
        self.model_def: Optional[LSTMModule] = None
        self.params: Any = None
        self.optimizer_state: Optional[optax.OptState] = None
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.feature_names = []

    def _create_sequences(self, data) -> Tuple[Any, Any]:
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    @staticmethod
    def _train_step(
        params,
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        batch_x: jax.Array,
        batch_y: jax.Array,
        rng: jax.Array,
        model_def: LSTMModule,
    ):
        """Single training step function using Flax (JIT compiled)."""
        # Create a new RNG key for this step
        rng, dropout_rng = jax.random.split(rng)

        # Define loss function operating on trainable parameters
        def loss_fn(params):
            # Apply the parameters to the model with dropout enabled
            predictions = model_def.apply({"params": params}, batch_x, training=True, rngs={"dropout": dropout_rng})
            mse_loss = jnp.mean((predictions - batch_y.squeeze()) ** 2)
            return mse_loss, predictions

        # Compute loss and gradients
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Compute parameter updates using optimizer
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state)

        # Apply updates to model parameters
        updated_params = optax.apply_updates(params, updates)

        metrics = {"loss": loss}
        return updated_params, new_optimizer_state, metrics, rng

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> "LSTMModel":
        """Fit the LSTM model to training data using NNX."""
        try:
            self.feature_names = X_train.columns.tolist()
            train_data = X_train.copy()
            train_data["target"] = y_train
            cols = ["target"] + [col for col in train_data.columns if col != "target"]
            train_data = train_data[cols]
            train_array = train_data.values
            X, y = self._create_sequences(train_array)

            if len(X) == 0 or len(y) == 0:
                logger.error("Not enough data to create sequences")
                self.is_fitted = False
                return self

            if self.model_def is None:
                input_features = X.shape[2]
                # Initialize the model definition
                self.model_def = LSTMModule(
                    input_features=input_features,
                    lstm_units=self.lstm_units,
                    dropout_rate=self.dropout_rate,
                )

                # Initialize model parameters
                self.rng, init_rng = jax.random.split(self.rng)
                variables = self.model_def.init(init_rng, jnp.ones((1, self.sequence_length, input_features)))

                # Extract just the params dict from the full variables dict
                self.params = variables["params"]

                if self.params is None:
                    raise ValueError("Failed to initialize parameters")

                # Initialize optimizer state
                self.optimizer_state = self.optimizer.init(self.params)

            if self.model_def is None or self.optimizer_state is None:
                logger.error("Model or optimizer state failed to initialize.")
                self.is_fitted = False
                return self

            num_batches = len(X) // self.batch_size
            patience = 25
            best_loss = float("inf")
            patience_counter = 0
            # Initialize best parameters
            best_params = None

            static_optimizer = self.optimizer

            for epoch in range(self.epochs):
                # Create a new RNG key for shuffling
                self.rng, shuffle_key = jax.random.split(self.rng)
                perm = jax.random.permutation(shuffle_key, len(X))

                X_shuffled = X[perm]
                y_shuffled = y[perm]

                epoch_loss = 0.0
                for batch in range(num_batches):
                    batch_start = batch * self.batch_size
                    batch_end = batch_start + self.batch_size
                    batch_x_np = X_shuffled[batch_start:batch_end]
                    batch_y_np = y_shuffled[batch_start:batch_end]

                    if len(batch_x_np) < self.batch_size:
                        continue

                    batch_x = jnp.array(batch_x_np)
                    batch_y = jnp.array(batch_y_np)

                    if self.optimizer_state is None:
                        logger.error("Optimizer state is None during training loop.")
                        self.is_fitted = False
                        return self

                    # Create JIT-compiled training step function
                    train_step_jit = jax.jit(self._train_step, static_argnames=["optimizer", "model_def"])

                    # Update parameters and optimizer state with training step
                    self.params, self.optimizer_state, metrics, self.rng = train_step_jit(
                        self.params, self.optimizer_state, static_optimizer, batch_x, batch_y, self.rng, self.model_def
                    )
                    epoch_loss += metrics["loss"]

                if num_batches > 0:
                    epoch_loss /= num_batches

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    # Save the best parameters
                    best_params = jax.tree_util.tree_map(lambda x: x, self.params)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

            # Use the best parameters found during training
            self.params = best_params
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}", exc_info=True)
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        """Generate predictions using the fitted LSTM model.

        Args:
            X: DataFrame with input features
            **kwargs: Additional keyword arguments for compatibility with BaseModel

        Returns:
            numpy array of predictions
        """
        if not self.is_fitted or self.model_def is None or self.params is None:
            logger.warning("LSTM model not fitted yet or model is None")
            return np.array([])

        try:
            # Prepare data for prediction
            pred_data = X.copy()
            if self.target_column not in pred_data.columns:
                pred_data.insert(0, "target", 0)
            else:
                cols = [self.target_column] + [col for col in pred_data.columns if col != self.target_column]
                pred_data = pred_data[cols]

            pred_array = pred_data.values

            if len(pred_array) < self.sequence_length:
                logger.warning(f"Not enough data for prediction (need {self.sequence_length}, got {len(pred_array)})")
                return np.array([])

            # Get the input feature dimension
            input_features = pred_array.shape[1]

            # Create a new model instance specifically for prediction
            # This ensures we have the correct input shape
            prediction_model = LSTMModule(
                input_features=input_features, lstm_units=self.lstm_units, dropout_rate=self.dropout_rate
            )

            # Initialize the prediction model with a dummy input of the correct shape
            self.rng, init_rng = jax.random.split(self.rng)
            dummy_input = jnp.ones((1, self.sequence_length, input_features))

            # Initialize parameters for the prediction model
            prediction_params = prediction_model.init(init_rng, dummy_input)

            # Extract the parameter structure but keep the trained values where possible
            # This is a key step to handle the shape mismatch
            def transfer_params(trained_params, new_params_struct, path=""):
                """Transfer parameters from trained model to new model where shapes match."""
                if isinstance(new_params_struct, dict):
                    result = {}
                    for k, v in new_params_struct.items():
                        new_path = f"{path}/{k}" if path else k
                        if k in trained_params:
                            result[k] = transfer_params(trained_params[k], v, new_path)
                        else:
                            result[k] = v
                    return result
                else:
                    # For leaf nodes (actual parameters), use trained values if shapes match
                    trained_leaf = jax.tree_util.tree_map(lambda x: x, trained_params)
                    if hasattr(trained_leaf, "shape") and hasattr(new_params_struct, "shape"):
                        if trained_leaf.shape == new_params_struct.shape:
                            return trained_leaf
                    return new_params_struct

            # Transfer parameters from trained model to prediction model
            try:
                adapted_params = transfer_params({"params": self.params}, prediction_params)["params"]
            except Exception as e:
                logger.warning(f"Parameter transfer failed: {str(e)}. Using initialized parameters.")
                adapted_params = prediction_params["params"]

            # Define prediction function with the new model
            def predict_step(params, seq_batch):
                return prediction_model.apply({"params": params}, seq_batch, training=False)

            # JIT-compile the prediction function
            predict_step_jit = jax.jit(predict_step)

            # Make predictions
            predictions_list = []
            for i in range(len(pred_array) - self.sequence_length + 1):
                seq_np = pred_array[i : i + self.sequence_length]
                seq_jax = jnp.array(seq_np[None, :, :])
                try:
                    # Check for NaN values in the input
                    if np.isnan(seq_jax).any():
                        logger.warning(f"Input sequence {i} contains NaN values, splicing timesteps")
                        # Find non-NaN timesteps
                        valid_timesteps = []
                        for t in range(seq_np.shape[0]):
                            if not np.isnan(seq_np[t]).any():
                                valid_timesteps.append(t)

                        if len(valid_timesteps) >= self.sequence_length // 2:  # At least half of timesteps are valid
                            # Create a new sequence with only valid timesteps
                            valid_seq = seq_np[valid_timesteps]
                            # If we don't have enough timesteps, repeat the last valid one
                            if len(valid_timesteps) < self.sequence_length:
                                padding = np.tile(valid_seq[-1:], (self.sequence_length - len(valid_timesteps), 1))
                                valid_seq = np.vstack([valid_seq, padding])
                            # Take the last self.sequence_length timesteps if we have more than needed
                            if len(valid_seq) > self.sequence_length:
                                valid_seq = valid_seq[-self.sequence_length :]
                            # Convert to JAX array
                            valid_seq_jax = jnp.array(valid_seq[None, :, :])
                            # Make prediction
                            pred = predict_step_jit(adapted_params, valid_seq_jax)
                            predictions_list.append(float(pred[0]))
                        else:
                            # Not enough valid timesteps, use last non-NaN value as fallback
                            last_val = 0.0
                            for t in range(seq_np.shape[0] - 1, -1, -1):
                                if not np.isnan(seq_np[t, 0]):
                                    last_val = seq_np[t, 0]
                                    break
                            predictions_list.append(float(last_val))
                    else:
                        # No NaN values, proceed normally
                        pred = predict_step_jit(adapted_params, seq_jax)
                        predictions_list.append(float(pred[0]))
                except Exception as e:
                    logger.warning(f"Prediction failed for sequence {i}: {str(e)}")
                    # Use a fallback prediction (last non-NaN value) if the model fails
                    last_val = 0.0
                    for t in range(seq_np.shape[0] - 1, -1, -1):
                        if not np.isnan(seq_np[t, 0]):
                            last_val = seq_np[t, 0]
                            break
                    predictions_list.append(float(last_val))

            num_predictions = len(predictions_list)
            num_expected = len(X)
            if num_predictions < num_expected:
                padding = np.full(num_expected - num_predictions, np.nan)
                final_predictions = np.concatenate((padding, np.array(predictions_list)))
            else:
                final_predictions = np.array(predictions_list[:num_expected])

            return final_predictions

        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {e}", exc_info=True)
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model state to disk."""
        directory = directory / "lstm"
        directory.mkdir(exist_ok=True, parents=True)
        model_dir = directory / f"lstm_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)

        params_path = model_dir / "params.pkl"
        model_path = model_dir / "model.pkl"
        config_path = model_dir / "config.pkl"

        try:
            if self.is_fitted and self.model_def is not None and self.params is not None:
                # Save the model definition and parameters using pickle
                with open(model_path, "wb") as f:
                    pickle.dump(self.model_def, f)
                with open(params_path, "wb") as f:
                    pickle.dump(self.params, f)

                save_config = {
                    "target_column": self.target_column,
                    "horizon": self.horizon,
                    "sequence_length": self.sequence_length,
                    "lstm_units": self.lstm_units,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "seed": self.seed,
                    "is_fitted": self.is_fitted,
                    "feature_names": self.feature_names,
                    "input_features": len(self.feature_names),
                }
                with open(config_path, "wb") as f:
                    pickle.dump(save_config, f)

                logger.info(f"LSTM model saved to {model_dir}")
            else:
                logger.warning("Cannot save LSTM model: not fitted or model is None")
            return model_dir
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return directory

    def load(self, model_path: Path) -> "LSTMModel":
        """Load model state from disk."""
        try:
            model_file = model_path / "model.pkl"
            params_file = model_path / "params.pkl"
            config_path = model_path / "config.pkl"

            if not model_file.exists() or not params_file.exists() or not config_path.exists():
                logger.error(f"Cannot load LSTM model: Missing files in {model_path}")
                self.is_fitted = False
                return self

            with open(config_path, "rb") as f:
                loaded_config = pickle.load(f)

            self.target_column = loaded_config["target_column"]
            self.horizon = loaded_config["horizon"]
            self.sequence_length = loaded_config["sequence_length"]
            self.lstm_units = loaded_config["lstm_units"]
            self.dropout_rate = loaded_config["dropout_rate"]
            self.learning_rate = loaded_config["learning_rate"]
            self.epochs = loaded_config["epochs"]
            self.batch_size = loaded_config["batch_size"]
            self.seed = loaded_config["seed"]
            self.is_fitted = loaded_config["is_fitted"]
            self.feature_names = loaded_config.get("feature_names", [])
            input_features = loaded_config.get("input_features")
            if input_features is None and self.feature_names:
                input_features = len(self.feature_names)
            if input_features is None:
                raise ValueError("Cannot determine input_features for model reconstruction from config.")

            # Initialize RNG
            self.rng = jax.random.key(self.seed)

            if self.is_fitted:
                # Load the model definition and parameters
                with open(model_file, "rb") as f:
                    self.model_def = pickle.load(f)
                with open(params_file, "rb") as f:
                    self.params = pickle.load(f)

                # Reinitialize optimizer only if params are loaded successfully
                if self.params is not None:
                    self.optimizer = optax.adam(learning_rate=self.learning_rate)
                    self.optimizer_state = self.optimizer.init(self.params)

                logger.info(f"LSTM model loaded from {model_path}")
            else:
                logger.warning(f"Model loaded from {model_path}, but was not marked as fitted.")
                self.model = None
                self.optimizer_state = None

            return self

        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}", exc_info=True)
            self.is_fitted = False
            self.model = None
            self.optimizer_state = None
            return self
