# src/models/lstm.py
"""LSTM model implementation using Flax/JAX for time series forecasting."""

import logging
import pickle

# Import necessary modules
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn  # Use standard Flax linen
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pandas as pd

# Removed TrainState and FrozenDict imports
from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LSTMCell(nn.Module):
    """LSTM cell implementation using standard Flax."""

    in_features: int
    hidden_features: int

    @nn.compact
    def __call__(self, carry, inputs):
        c, h = carry

        # Combine inputs and hidden state
        concat_input = jnp.concatenate([inputs, h], axis=-1)

        # Apply gates
        gates = nn.Dense(features=4 * self.hidden_features)(concat_input)
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        # Apply activations
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        # Update cell state
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)

        return (new_c, new_h), new_h

    def initialize_carry(self, batch_dims):
        """Initialize the cell state and hidden state."""
        return (jnp.zeros(batch_dims + (self.hidden_features,)), jnp.zeros(batch_dims + (self.hidden_features,)))


class LSTMModule(nn.Module):
    """LSTM neural network module using Flax."""

    input_features: int
    lstm_units: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, *, training: bool = False):
        """Forward pass through the LSTM module."""
        batch_size = x.shape[0]

        # First LSTM layer
        lstm_cell1 = LSTMCell(in_features=self.input_features, hidden_features=self.lstm_units)
        init_carry1 = lstm_cell1.initialize_carry((batch_size,))

        # Scan over sequence dimension
        (_, _), outputs1 = nn.scan(
            lstm_cell1,
            variable_broadcast="params",
            # Don't split RNGs across scan steps
            in_axes=1,
            out_axes=1,
        )(init_carry1, x)

        # Apply dropout after first LSTM layer
        outputs1 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(outputs1)

        # Second LSTM layer
        lstm_cell2 = LSTMCell(in_features=self.lstm_units, hidden_features=self.lstm_units // 2)
        init_carry2 = lstm_cell2.initialize_carry((batch_size,))

        # Scan over sequence dimension
        (_, _), outputs2 = nn.scan(
            lstm_cell2,
            variable_broadcast="params",
            # Don't split RNGs across scan steps
            in_axes=1,
            out_axes=1,
        )(init_carry2, outputs1)

        # Apply dropout after second LSTM layer
        outputs2 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(outputs2)

        # Get the last output from the sequence
        last_output = outputs2[:, -1, :]

        # Apply final dense layer
        y = nn.Dense(features=1)(last_output)
        return y.squeeze(-1)


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting using JAX/Flax NNX."""

    optimizer_state: Optional[optax.OptState] = None
    model_def: Optional[LSTMModule] = None
    params: Optional[Any] = None

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
        self.optimizer_state = None
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
    @jax.jit
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
        rng, _ = jax.random.split(rng)

        # Define loss function operating on trainable parameters
        def loss_fn(params):
            # Apply the parameters to the model
            # Apply the parameters to the model with dropout enabled
            predictions = model_def.apply({"params": params}, batch_x, training=True)
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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LSTMModel":
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
                    input_features=input_features, lstm_units=self.lstm_units, dropout_rate=self.dropout_rate
                )

                # Initialize model parameters
                self.rng, init_rng = jax.random.split(self.rng)
                self.params = self.model_def.init(init_rng, jnp.ones((1, self.sequence_length, input_features)))

                # Extract just the params dict from the full variables dict
                self.params = self.params["params"]

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
            best_params = self.params

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

                    # Update model and optimizer state with training step
                    # Update parameters and optimizer state with training step
                    self.params, self.optimizer_state, metrics, self.rng = self._train_step(
                        self.params, self.optimizer_state, static_optimizer, batch_x, batch_y, self.rng, self.model_def
                    )
                    epoch_loss += metrics["loss"]

                if num_batches > 0:
                    epoch_loss /= num_batches

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_params = self.params
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
            logger.info("LSTM NNX model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting LSTM NNX model: {e}", exc_info=True)
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame):
        """Generate predictions using the fitted LSTM NNX model."""
        if not self.is_fitted or self.model_def is None or self.params is None:
            logger.warning("LSTM NNX model not fitted yet or model is None")
            return np.array([])

        try:
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

            # Define a JIT-compiled prediction function
            @jax.jit
            def predict_step(params, model_def, seq_batch):
                # Use training=False for prediction mode (disables dropout)
                return model_def.apply({"params": params}, seq_batch, training=False)

            predictions_list = []

            for i in range(len(pred_array) - self.sequence_length + 1):
                seq_np = pred_array[i : i + self.sequence_length]
                seq_jax = jnp.array(seq_np[None, :, :])
                pred = predict_step(self.params, self.model_def, seq_jax)
                predictions_list.append(float(pred[0]))

            num_predictions = len(predictions_list)
            num_expected = len(X)
            if num_predictions < num_expected:
                padding = np.full(num_expected - num_predictions, np.nan)
                final_predictions = np.concatenate((padding, np.array(predictions_list)))
            else:
                final_predictions = np.array(predictions_list[:num_expected])

            return final_predictions

        except Exception as e:
            logger.error(f"Error predicting with LSTM NNX model: {e}", exc_info=True)
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model state to disk using NNX."""
        directory = directory / "lstm_nnx"
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

                logger.info(f"LSTM NNX model saved to {model_dir}")
            else:
                logger.warning("Cannot save LSTM NNX model: not fitted or model is None")
            return model_dir
        except Exception as e:
            logger.error(f"Error saving LSTM NNX model: {e}")
            return directory

    def load(self, model_path: Path) -> "LSTMModel":
        """Load model state from disk using NNX."""
        try:
            model_file = model_path / "model.pkl"
            params_file = model_path / "params.pkl"
            config_path = model_path / "config.pkl"

            if not model_file.exists() or not params_file.exists() or not config_path.exists():
                logger.error(f"Cannot load LSTM NNX model: Missing files in {model_path}")
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

                logger.info(f"LSTM NNX model loaded from {model_path}")
            else:
                logger.warning(f"Model loaded from {model_path}, but was not marked as fitted.")
                self.model = None
                self.optimizer_state = None

            return self

        except Exception as e:
            logger.error(f"Error loading LSTM NNX model: {e}", exc_info=True)
            self.is_fitted = False
            self.model = None
            self.optimizer_state = None
            return self
