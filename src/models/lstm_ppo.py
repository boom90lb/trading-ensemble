# src/models/lstm_ppo.py
"""LSTM-PPO model implementation for reinforcement learning trading."""

import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.config import MODELS_DIR
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM feature extractor for processing sequential market data."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
    ):
        """Initialize the LSTM feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            lstm_hidden_size: Size of LSTM hidden layer
            lstm_layers: Number of LSTM layers
        """
        super().__init__(observation_space, features_dim)

        # Input dimension from observation space
        n_inputs = int(np.prod(observation_space.shape))

        # LSTM for sequential data processing
        self.lstm = nn.LSTM(
            input_size=n_inputs, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True
        )

        # Final linear layer to get the feature dimension
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor.

        Args:
            observations: Input observations [batch_size, seq_len, n_features]

        Returns:
            Extracted features [batch_size, features_dim]
        """
        # Reshape if necessary (if not sequential input)
        batch_size = observations.size(0)
        if len(observations.shape) == 2:
            # Convert [batch_size, features] to [batch_size, 1, features]
            observations = observations.unsqueeze(1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(observations)

        # We only need the last output
        last_output = lstm_out[:, -1, :]

        # Pass through linear layer and activation
        features = self.linear(last_output)
        features = self.relu(features)

        return features


class LSTMPolicy(ActorCriticPolicy):
    """Custom policy that uses an LSTM feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        """Initialize the LSTM policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Pass custom feature extractor
        kwargs["features_extractor_class"] = LSTMFeatureExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=64, lstm_hidden_size=128, lstm_layers=2)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)


class TradingEnvironment(gym.Env):
    """Custom gym environment for trading with reinforcement learning."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        window_size: int = 20,
        max_trade_duration: int = 10,
        transaction_cost: float = 0.001,
        initial_balance: float = 10000.0,
        reward_scaling: float = 1.0,
    ):
        """Initialize the trading environment.

        Args:
            df: DataFrame with price data and features
            features: List of feature columns to use
            window_size: Size of observation window
            max_trade_duration: Maximum number of steps for a single trade
            transaction_cost: Transaction cost ratio
            initial_balance: Initial account balance
            reward_scaling: Scaling factor for rewards
        """
        super().__init__()

        # Store parameters
        self.df = df
        self.features = features
        self.window_size = window_size
        self.max_trade_duration = max_trade_duration
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling

        # Validate features
        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Feature {feature} not in DataFrame")

        # Environment state - initialized in reset
        self.current_step: Optional[int] = None
        self.current_position: Optional[float] = None
        self.current_balance: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.entry_step: Optional[int] = None
        self.done: Optional[bool] = None

        # Action and observation spaces
        # Action space: [position_size] from -1.0 (full short) to 1.0 (full long)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: normalized features over window_size steps
        num_features = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, num_features), dtype=np.float32
        )

        # Initialize - call reset to properly set initial state
        # self.reset() # Reset is typically called externally before starting training/evaluation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Set the starting point of the episode
        if options and "start_idx" in options:
            start_idx = options["start_idx"]
        else:
            # Start after window_size to have enough history
            start_idx = self.window_size

        # Ensure start_idx is valid
        self.current_step = max(self.window_size, min(start_idx, len(self.df) - 1))

        # Reset state
        self.current_position = 0.0  # No position
        self.current_balance = self.initial_balance
        self.entry_price = 0.0
        self.entry_step = 0
        self.done = False

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: Action to take (position size)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            # Return last state if already done
            logger.warning("Called step() on a done environment. Returning last observation.")
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, False, info

        # Ensure current_step is valid
        if self.current_step is None or self.current_step >= len(self.df):
            raise RuntimeError("Invalid environment state: current_step is invalid or out of bounds.")

        # Extract action (position size)
        position_size = float(action[0])  # Range: -1.0 to 1.0

        # Get current price
        current_price = self.df.iloc[self.current_step]["close"]

        # Calculate reward based on position change
        reward = self._calculate_reward(position_size, current_price)

        # Update state
        self.current_position = position_size

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = False
        truncated = False

        # Episode ends when we reach the end of data
        if self.current_step >= len(self.df):
            terminated = True
            self.done = True
            # Adjust step back to last valid index for final observation/info
            self.current_step = len(self.df) - 1

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get the current observation.

        Returns:
            Observation as numpy array
        """
        if self.current_step is None:
            # Should not happen if reset is called first
            return np.zeros((self.window_size, len(self.features)), dtype=np.float32)

        # Extract window of features
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1

        # Get feature data
        observations = self.df.iloc[start_idx:end_idx][self.features].values

        # Ensure the shape is correct by padding if needed
        if observations.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - observations.shape[0], len(self.features)))
            observations = np.concatenate([padding, observations])
        elif observations.shape[0] > self.window_size:
            # This case might happen if current_step was adjusted after termination
            observations = observations[-self.window_size :]

        return observations.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get additional information.

        Returns:
            Dictionary with additional information
        """
        if self.current_step is None or self.current_step >= len(self.df):
            price = 0  # Or handle appropriately
        else:
            price = self.df.iloc[self.current_step]["close"]

        return {
            "balance": self.current_balance,
            "position": self.current_position,
            "step": self.current_step,
            "price": price,
        }

    def _calculate_reward(self, new_position: float, current_price: float) -> float:
        """Calculate reward based on position change.

        Args:
            new_position: New position size (-1.0 to 1.0)
            current_price: Current price

        Returns:
            Reward value
        """
        prev_position = self.current_position if self.current_position is not None else 0.0
        entry_price = self.entry_price if self.entry_price is not None else 0.0

        # If this is the first step or position hasn't changed, no immediate reward
        if prev_position == new_position:
            reward = 0.0
        else:
            # Calculate transaction cost for position change
            position_change = abs(new_position - prev_position)
            transaction_cost = position_change * current_price * self.transaction_cost

            # Update balance based on transaction cost
            if self.current_balance is not None:
                self.current_balance -= transaction_cost
            else:
                self.current_balance = self.initial_balance - transaction_cost

            # If closing a position, calculate profit/loss
            if (prev_position > 0 and new_position <= 0) or (prev_position < 0 and new_position >= 0):
                # Calculate profit/loss
                if prev_position > 0:  # Long position
                    pnl = prev_position * (current_price - entry_price)
                else:  # Short position
                    pnl = -prev_position * (entry_price - current_price)

                # Update balance
                if self.current_balance is not None:
                    self.current_balance += pnl
                # else: handle error or initialization state

                # Reward is the PnL minus transaction cost
                reward = pnl - transaction_cost
            else:
                # Just opened or adjusted position, reward is negative transaction cost
                reward = -transaction_cost

            # Update entry price for new position
            if new_position != 0:
                self.entry_price = current_price
                self.entry_step = self.current_step

        # Scale reward
        reward *= self.reward_scaling

        return reward


class LSTMPPO(BaseModel):
    """LSTM-PPO reinforcement learning model for trading."""

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 1,
        features: Optional[List[str]] = None,
        window_size: int = 20,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        seed: int = 42,
        device: str = "auto",
        **kwargs,
    ):
        """Initialize the LSTM-PPO model.

        Args:
            target_column: Target column to predict
            horizon: Forecast horizon
            features: List of feature columns to use
            window_size: Size of observation window
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Minibatch size for updates
            n_epochs: Number of epochs for update
            gamma: Discount factor
            gae_lambda: Factor for GAE
            clip_range: PPO clip range
            transaction_cost: Transaction cost ratio
            reward_scaling: Scaling factor for rewards
            seed: Random seed
            device: Device to run on ('auto', 'cpu', 'cuda')
            **kwargs: Additional model parameters
        """
        super().__init__(name="lstm_ppo", target_column=target_column, horizon=horizon, **kwargs)

        # Store parameters
        self.features = features or ["open", "high", "low", "close", "volume"]
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.seed = seed
        self.device = device

        # RL environment and model
        self.env: Optional[TradingEnvironment] = None
        self.model: Optional[PPO] = None

    def _create_environment(self, df: pd.DataFrame) -> TradingEnvironment:
        """Create a trading environment.

        Args:
            df: DataFrame with price data and features

        Returns:
            Trading environment
        """
        return TradingEnvironment(
            df=df,
            features=self.features,
            window_size=self.window_size,
            transaction_cost=self.transaction_cost,
            reward_scaling=self.reward_scaling,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "LSTMPPO":
        """Fit the LSTM-PPO model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters (e.g., total_timesteps)

        Returns:
            Self for method chaining
        """
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()

            # Combine features and target into a single DataFrame
            train_df = X_train.copy()
            train_df[self.target_column] = y_train

            # Create the environment
            self.env = self._create_environment(train_df)

            # Create the LSTM-PPO model
            self.model = PPO(  # type: ignore
                policy=LSTMPolicy,
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                seed=self.seed,
                device=self.device,
                verbose=1,
            )

            # Get training iterations from kwargs or use default
            total_timesteps = kwargs.get("total_timesteps", 100000)

            # Train the model
            self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

            self.is_fitted = True
            logger.info("LSTM-PPO model trained")

        except Exception as e:
            logger.error(f"Error training LSTM-PPO model: {e}")
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (actions) using the fitted LSTM-PPO model.

        Args:
            X: Features

        Returns:
            Array of predictions (actions)
        """
        if not self.is_fitted or self.model is None:
            logger.warning("LSTM-PPO model not fitted yet")
            return np.array([])

        try:
            # Create a test environment
            test_df = X.copy()

            # If target column is missing, add a dummy one
            if self.target_column not in test_df.columns:
                test_df[self.target_column] = 0.0

            test_env = self._create_environment(test_df)

            # Get initial observation
            obs, _ = test_env.reset()

            # Generate predictions for each step
            predictions = []
            current_obs = obs

            for _ in range(len(test_df)):
                # Ensure current_obs is correctly shaped for predict
                if len(current_obs.shape) == 2:
                    # Reshape [seq_len, features] to [1, seq_len, features]
                    current_obs = np.expand_dims(current_obs, axis=0)
                elif len(current_obs.shape) != 3 or current_obs.shape[0] != 1:
                    logger.error(f"Invalid observation shape for prediction: {current_obs.shape}")
                    return np.array([])  # Or handle error appropriately

                # Get action from model
                action, _ = self.model.predict(current_obs, deterministic=True)

                # Store prediction (action)
                predictions.append(action[0])

                # Step environment to get next observation for the *next* prediction
                # We pass the *predicted* action to the environment step
                next_obs, _, terminated, truncated, _ = test_env.step(action)
                current_obs = next_obs

                # Check for termination or truncation
                if terminated or truncated:
                    break

            # Pad predictions if the loop broke early
            if len(predictions) < len(X):
                padding = np.zeros(len(X) - len(predictions))
                predictions.extend(padding)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error predicting with LSTM-PPO model: {e}")
            return np.array([])

    def get_policy(self) -> Optional[ActorCriticPolicy]:
        """Get the trained policy.

        Returns:
            Trained policy or None if not fitted
        """
        if not self.is_fitted or self.model is None:
            logger.warning("LSTM-PPO model not fitted yet")
            return None

        return self.model.policy  # type: ignore

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save the model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model directory
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)

        # Create model directory
        model_dir = directory / f"{self.name}_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)

        # Save the model
        if self.is_fitted and self.model is not None:
            model_path = model_dir / "model.zip"
            self.model.save(model_path)
            logger.info(f"LSTM-PPO model saved to {model_path}")
        else:
            logger.warning("LSTM-PPO model not saved: model is not fitted.")

        # Save parameters (optional, but good practice)
        params_path = model_dir / "params.pkl"
        params = {
            "target_column": self.target_column,
            "horizon": self.horizon,
            "features": self.features,
            "window_size": self.window_size,
            "learning_rate": self.learning_rate,
            # Include other relevant hyperparameters if needed
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        return model_dir

    def load(self, model_path: Path) -> "LSTMPPO":
        """Load model from disk.

        Args:
            model_path: Path to the saved model zip file (e.g., .../model.zip)

        Returns:
            Loaded model
        """
        try:
            # Load parameters first if they exist
            params_path = model_path.parent / "params.pkl"
            if params_path.exists():
                with open(params_path, "rb") as f:
                    params = pickle.load(f)
                # Restore parameters (optional but good practice)
                self.target_column = params.get("target_column", self.target_column)
                self.horizon = params.get("horizon", self.horizon)
                self.features = params.get("features", self.features)
                self.window_size = params.get("window_size", self.window_size)
                self.learning_rate = params.get("learning_rate", self.learning_rate)
                self.feature_names = params.get("feature_names", [])
                # is_fitted will be set after successful model load

            # Create a dummy environment required for loading
            # Ensure features and target_column are set before creating env
            dummy_df = pd.DataFrame({feature: np.zeros(self.window_size + 1) for feature in self.features})
            dummy_df[self.target_column] = 0.0
            self.env = self._create_environment(dummy_df)

            # Load the model
            self.model = PPO.load(model_path, env=self.env, device=self.device)  # type: ignore
            self.is_fitted = True

            logger.info(f"LSTM-PPO model loaded from {model_path}")

            return self
        except Exception as e:
            logger.error(f"Error loading LSTM-PPO model from {model_path}: {e}")
            self.is_fitted = False
            raise
