# src/models/lstm_ppo.py
"""LSTM-PPO model implementation using JAX/Flax for reinforcement learning trading."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# import chex  # Uncomment if needed for future development
import distrax  # type: ignore
import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pandas as pd
from gymnasium import spaces

from src.config import MODELS_DIR
from src.models.base import BaseModel

# Callable imported but not used - kept for potential future use


logger = logging.getLogger(__name__)


class LSTMFeatureExtractor(nnx.Module):
    """LSTM feature extractor for processing sequential market data using NNX."""

    def __init__(self, *, input_features: int, features_dim: int, lstm_hidden_size: int, rngs: nnx.Rngs):
        """Initialize NNX LSTM Feature Extractor."""
        self.lstm_cell = nnx.LSTMCell(in_features=input_features, hidden_features=lstm_hidden_size, rngs=rngs)
        self.dense = nnx.Linear(in_features=lstm_hidden_size, out_features=features_dim, rngs=rngs)

        # Note: Scan implementation is planned for future updates
        # This is a placeholder for the scan body that will be used in the future
        # def _scan_body(cell: nnx.LSTMCell, carry, xs):
        #     new_carry, hidden = cell(carry, xs)
        #     return new_carry, hidden

        # Use the updated nnx.Scan signature
        # Create a simple LSTM cell first
        self.lstm_cell = nnx.LSTMCell(input_features, lstm_hidden_size, rngs=nnx.Rngs(0))

        # For now, we'll use a simpler approach without scan
        # This will be updated in a future PR to use the proper scan API

    def __call__(self, observations: jax.Array) -> jax.Array:
        """Forward pass through the feature extractor."""
        if observations.ndim == 2:
            observations = observations[:, None, :]

        batch_size, _, _ = observations.shape

        init_carry = self.lstm_cell.initialize_carry((batch_size,))
        observations_scanned = jnp.transpose(observations, (1, 0, 2))

        # Process each timestep manually for now
        # This is a temporary solution until we properly implement scan
        hidden_states = []
        carry = init_carry

        for i in range(observations_scanned.shape[0]):
            carry, hidden = self.lstm_cell(carry, observations_scanned[i])
            hidden_states.append(hidden)

        # Convert to array - ensure proper stacking of arrays
        hidden_states_array = jnp.stack(hidden_states)

        # Use the last hidden state from the array
        last_hidden = hidden_states_array[-1]
        x = self.dense(last_hidden)
        x = nnx.relu(x)
        return x


class ActorCritic(nnx.Module):
    """Actor-Critic network with NNX LSTM feature extractor."""

    action_dim: int
    input_features: int
    features_dim: int
    lstm_hidden_size: int

    def __init__(
        self,
        *,
        input_features: int,
        action_dim: int,
        features_dim: int = 64,
        lstm_hidden_size: int = 128,
        rngs: nnx.Rngs,
    ):
        """Initialize the NNX Actor-Critic network."""
        self.action_dim = action_dim
        self.input_features = input_features
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size

        self.feature_extractor = LSTMFeatureExtractor(
            input_features=self.input_features,
            features_dim=self.features_dim,
            lstm_hidden_size=self.lstm_hidden_size,
            rngs=rngs,
        )

        self.actor_mean = nnx.Linear(in_features=self.features_dim, out_features=self.action_dim, rngs=rngs)
        self.actor_logstd = nnx.Param(jnp.zeros(self.action_dim))
        self.critic_value = nnx.Linear(in_features=self.features_dim, out_features=1, rngs=rngs)

    def __call__(
        self, observations: jax.Array, *, rngs: Optional[nnx.Rngs] = None
    ) -> Tuple[distrax.Distribution, jax.Array]:
        # rngs parameter is required by the NNX API but not used directly in this method
        """Forward pass through the Actor-Critic network."""
        features = self.feature_extractor(observations)
        mean = self.actor_mean(features)
        log_std = self.actor_logstd.value
        std = jnp.exp(log_std)
        pi = distrax.MultivariateNormalDiag(mean, std)

        value = self.critic_value(features)
        return pi, jnp.squeeze(value, axis=-1)


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

        self.df = df
        self.features = features
        self.window_size = window_size
        self.max_trade_duration = max_trade_duration
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling

        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Feature {feature} not in DataFrame")

        self.current_step: Optional[int] = None
        self.current_position: Optional[float] = None
        self.current_balance: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.entry_step: Optional[int] = None
        self.done: Optional[bool] = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        num_features = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, num_features), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if options and "start_idx" in options:
            start_idx = options["start_idx"]
        else:
            start_idx = self.window_size

        self.current_step = max(self.window_size, min(start_idx, len(self.df) - 1))

        self.current_position = 0.0
        self.current_balance = self.initial_balance
        self.entry_price = 0.0
        self.entry_step = 0
        self.done = False

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
            logger.warning("Called step() on a done environment. Returning last observation.")
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, False, info

        if self.current_step is None or self.current_step >= len(self.df):
            raise RuntimeError("Invalid environment state: current_step is invalid or out of bounds.")

        position_size = float(action[0])
        current_price = self.df.iloc[self.current_step]["close"]
        reward = self._calculate_reward(position_size, current_price)
        self.current_position = position_size
        self.current_step += 1

        terminated = False
        truncated = False
        if self.current_step >= len(self.df):
            terminated = True
            self.done = True
            self.current_step = len(self.df) - 1

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get the current observation.

        Returns:
            Observation as numpy array
        """
        if self.current_step is None:
            return np.zeros((self.window_size, len(self.features)), dtype=np.float32)

        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        observations = self.df.iloc[start_idx:end_idx][self.features].values

        if observations.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - observations.shape[0], len(self.features)))
            observations = np.concatenate([padding, observations])
        elif observations.shape[0] > self.window_size:
            observations = observations[-self.window_size :]

        return observations.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get additional information.

        Returns:
            Dictionary with additional information
        """
        if self.current_step is None or self.current_step >= len(self.df):
            price = 0
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

        if prev_position == new_position:
            reward = 0.0
        else:
            position_change = abs(new_position - prev_position)
            transaction_cost = position_change * current_price * self.transaction_cost

            if self.current_balance is not None:
                self.current_balance -= transaction_cost
            else:
                self.current_balance = self.initial_balance - transaction_cost

            if (prev_position > 0 and new_position <= 0) or (prev_position < 0 and new_position >= 0):
                if prev_position > 0:
                    pnl = prev_position * (current_price - entry_price)
                else:
                    pnl = -prev_position * (entry_price - current_price)

                if self.current_balance is not None:
                    self.current_balance += pnl

                reward = pnl - transaction_cost
            else:
                reward = -transaction_cost

            if new_position != 0:
                self.entry_price = current_price
                self.entry_step = self.current_step

        reward *= self.reward_scaling
        return reward


class PPOTrainer:
    """PPO trainer implementation using JAX/Flax NNX."""

    model_state: Optional[nnx.State] = None
    optimizer_state: Optional[optax.OptState] = None
    graph_def: Optional[nnx.GraphDef] = None
    rngs: nnx.Rngs
    optimizer: optax.GradientTransformation

    def __init__(
        self,
        env: TradingEnvironment,
        network_kwargs: Dict[str, Any],
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        seed: int = 42,
        device: str = "gpu",  # device parameter kept for API compatibility
    ):
        """Initialize the PPO trainer with NNX."""
        self.env = env
        self.network_kwargs = network_kwargs
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.seed = seed

        self.rngs = nnx.Rngs(params=seed, sample=seed + 1)
        self.optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=self.learning_rate))

        self._initialize_model()

    def _initialize_model(self):
        """Initialize NNX model components."""
        if self.env is None:
            raise ValueError("Environment is not set.")

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise ValueError("Currently only supports Box observation and action spaces")

        obs_shape = obs_space.shape
        act_shape = action_space.shape
        input_features = obs_shape[-1]
        action_dim = act_shape[0]

        network_instance = ActorCritic(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )

        # Use nnx.split instead of instance method
        graph_def, state = nnx.split(network_instance)
        self.graph_def = graph_def
        # Store the state and graph_def
        # Convert state to the expected type if needed
        # This ensures compatibility with the type annotations in the class
        # Cast to the expected type to satisfy type checker
        self.model_state = state  # type: ignore

        # For optimizer, we'll use a dummy empty dict for now
        # This is a temporary solution until we properly implement the optimizer
        self.optimizer_state = {}

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def",))
    def _get_action_and_value_jit(
        graph_def: nnx.GraphDef, model_state: nnx.State, observations: jax.Array, rngs: nnx.Rngs
    ):
        """JITted function to get action and value using NNX."""
        # Correct RNG splitting
        # Create a new RNG for sampling
        key = jax.random.key(0)  # Use a fixed key for deterministic behavior
        model_rngs = nnx.Rngs(params=key)
        # Use nnx.merge instead of instance method
        model = nnx.merge(graph_def, model_state)

        pi, value = model(observations, rngs=model_rngs)
        # Sample actions using JAX RNG
        actions = pi.sample(seed=jax.random.key(0))
        log_probs = pi.log_prob(actions)

        # Return original RNGs (implicitly consumed)
        return actions, log_probs, value, rngs

    def _get_action_and_value(self, observations):
        """Get action and value from the network."""
        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Model state/graph_def not initialized.")

        observations_jax = jnp.asarray(observations)
        actions, log_probs, value, _ = self._get_action_and_value_jit(
            self.graph_def, self.model_state, observations_jax, self.rngs
        )
        # Create a new RNG for next call
        self.rngs = nnx.Rngs(params=self.seed)
        return np.array(actions), np.array(log_probs), np.array(value)

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute generalized advantage estimation."""
        n_steps = len(rewards)
        advantages = jnp.zeros_like(rewards)
        last_gae_lam = 0
        values_extended = jnp.append(values, next_value)
        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - dones[t]
            current_val = values_extended[t]
            next_val = values_extended[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - current_val
            advantages = advantages.at[t].set(delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam)
            last_gae_lam = advantages[t]
        returns = advantages + values
        return returns, advantages

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "clip_range"))
    def _ppo_loss_jit(graph_def: nnx.GraphDef, model_state: nnx.State, batch: Dict, clip_range: float, rngs: nnx.Rngs):
        """JITted PPO loss function using NNX."""
        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Create a new RNG
        model_rngs = nnx.Rngs(params=0)
        # Use nnx.merge instead of instance method
        model = nnx.merge(graph_def, model_state)
        pi, values = model(observations, rngs=model_rngs)
        log_probs = pi.log_prob(actions)
        entropy = pi.entropy().mean()

        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -jnp.minimum(surrogate1, surrogate2).mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": 0.5 * ((log_probs - old_log_probs) ** 2).mean(),
        }
        return total_loss, info, rngs

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "clip_range", "optimizer"))
    def _update_step_jit(
        graph_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer_state: optax.OptState,
        batch: Dict,
        clip_range: float,
        optimizer: optax.GradientTransformation,
        rngs: nnx.Rngs,
    ):
        """JITted PPO update step using NNX."""
        step_rngs = rngs.fork()

        # Loss function operates on params state
        def loss_for_grad(params_state: nnx.State):
            # Need to reconstruct the *full* current state by merging dynamic params
            # into the static non-params state derived from the original model_state.
            # This requires passing the non-params state if it exists.
            # Assuming only Params are dynamic for now, merge into an empty state? No.
            # Pass the original model_state and merge params into it.
            current_state = model_state.merge(params_state)
            loss, info, _ = PPOTrainer._ppo_loss_jit(graph_def, current_state, batch, clip_range, step_rngs)
            return loss, info

        params_state = model_state.filter(nnx.Param)
        (loss, info), grads = nnx.value_and_grad(loss_for_grad, has_aux=True)(params_state)

        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params_state)
        # Apply updates to the model state using nnx.update
        # Create a new model state by merging the updates
        new_model_state = model_state  # Start with the original state
        nnx.update(new_model_state, updates)  # Apply updates in-place

        return new_model_state, new_optimizer_state, loss, info, rngs

    def _update_policy(self, rollout_data: Dict[str, jnp.ndarray]):
        """Update policy using collected rollout data."""
        if self.model_state is None or self.optimizer_state is None or self.graph_def is None:
            raise RuntimeError("Trainer state components not initialized before update.")

        n_samples = rollout_data["observations"].shape[0]
        indices = jnp.arange(n_samples)
        last_info = {}

        graph_def_static = self.graph_def
        optimizer_static = self.optimizer
        clip_range_static = self.clip_range

        for _ in range(self.n_epochs):
            # Need a JAX key for permutation
            perm_key = self.rngs["params"]()
            indices = jax.random.permutation(perm_key, indices)
            # No need to fork self.rngs here, perm_key consumption handled by JAX

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                batch = {k: v[batch_indices] for k, v in rollout_data.items()}

                # JIT call consumes rngs fork internally, returns original
                new_model_state, new_optimizer_state, _, last_info, _ = self._update_step_jit(
                    graph_def_static,
                    self.model_state,
                    self.optimizer_state,
                    batch,
                    clip_range_static,
                    optimizer_static,
                    self.rngs,
                )
                self.model_state = new_model_state
                self.optimizer_state = new_optimizer_state
                # Create a new RNG for next call
                self.rngs = nnx.Rngs(params=self.seed)

        return self.model_state, self.optimizer_state, last_info

    def learn(self, total_timesteps: int) -> Tuple[nnx.State, nnx.GraphDef]:
        """Train the PPO agent."""
        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Trainer state not initialized before learning.")

        obs_np, _ = self.env.reset()
        obs: jnp.ndarray = jnp.array(obs_np)

        buffer_size = self.n_steps
        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape
        if not isinstance(obs_shape, tuple) or not obs_shape:
            raise ValueError(f"Invalid observation space shape: {obs_shape}")
        if not isinstance(act_shape, tuple) or not act_shape:
            raise ValueError(f"Invalid action space shape: {act_shape}")

        rollout_buffer = {
            "observations": jnp.zeros((buffer_size,) + obs_shape, dtype=self.env.observation_space.dtype),
            "actions": jnp.zeros((buffer_size,) + act_shape, dtype=self.env.action_space.dtype),
            "rewards": jnp.zeros(buffer_size, dtype=jnp.float32),
            "dones": jnp.zeros(buffer_size, dtype=jnp.float32),
            "values": jnp.zeros(buffer_size, dtype=jnp.float32),
            "log_probs": jnp.zeros(buffer_size, dtype=jnp.float32),
        }
        step_count = 0
        n_updates = total_timesteps // self.n_steps

        for update in range(n_updates):
            if self.model_state is None or self.graph_def is None:
                raise RuntimeError("Model state became None during update loop.")

            for _ in range(self.n_steps):
                obs_batch = obs[None, :]
                action_np, log_prob_np, value_np = self._get_action_and_value(obs_batch)

                current_index = step_count % buffer_size
                rollout_buffer["observations"] = rollout_buffer["observations"].at[current_index].set(obs)
                rollout_buffer["actions"] = rollout_buffer["actions"].at[current_index].set(action_np.squeeze())
                rollout_buffer["values"] = rollout_buffer["values"].at[current_index].set(value_np.squeeze())
                rollout_buffer["log_probs"] = rollout_buffer["log_probs"].at[current_index].set(log_prob_np.squeeze())

                next_obs, reward, terminated, truncated, _ = self.env.step(action_np.squeeze())
                done = terminated or truncated

                rollout_buffer["rewards"] = rollout_buffer["rewards"].at[current_index].set(reward)
                rollout_buffer["dones"] = rollout_buffer["dones"].at[current_index].set(float(done))

                obs_jax = jnp.array(next_obs)
                if done:
                    obs_np, _ = self.env.reset()
                    obs_jax = jnp.array(obs_np)
                obs = obs_jax
                step_count += 1

            last_obs_batch = obs[None, :]
            _, _, last_value_np = self._get_action_and_value(last_obs_batch)
            returns, advantages = self._compute_gae(
                rollout_buffer["rewards"], rollout_buffer["values"], rollout_buffer["dones"], last_value_np.squeeze()
            )

            update_data = {
                "observations": rollout_buffer["observations"],
                "actions": rollout_buffer["actions"],
                "log_probs": rollout_buffer["log_probs"],
                "returns": returns,
                "advantages": advantages,
            }
            update_data["advantages"] = (update_data["advantages"] - update_data["advantages"].mean()) / (
                update_data["advantages"].std() + 1e-8
            )

            self.model_state, self.optimizer_state, update_info = self._update_policy(update_data)

            if update % 10 == 0:
                loss_val = update_info.get("policy_loss", jnp.nan)
                logger.info(f"Update {update}/{n_updates}, Loss: {float(loss_val):.4f}")

        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Model state or graph_def is None after training.")

        return self.model_state, self.graph_def


class LSTMPPO(BaseModel):
    """LSTM-PPO reinforcement learning model for trading using JAX/Flax NNX."""

    model_state: Optional[nnx.State] = None
    graph_def: Optional[nnx.GraphDef] = None
    trainer: Optional[PPOTrainer] = None
    env: Optional[TradingEnvironment] = None
    rngs: Optional[nnx.Rngs] = None

    def __init__(
        self,
        target_column: str = "close",
        horizon: int = 1,
        features: Optional[List[str]] = None,
        window_size: int = 20,
        features_dim: int = 64,
        lstm_hidden_size: int = 128,
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
        device: str = "gpu",  # device parameter kept for API compatibility
        **kwargs,
    ):
        """Initialize the LSTM-PPO model using NNX."""
        super().__init__(name="lstm_ppo", target_column=target_column, horizon=horizon, **kwargs)

        self.features = features or ["open", "high", "low", "close", "volume"]
        self.window_size = window_size
        self.network_kwargs = {
            "features_dim": features_dim,
            "lstm_hidden_size": lstm_hidden_size,
        }
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

        self.env = None
        self.model_state = None
        self.graph_def = None
        self.trainer = None
        self.rngs = None

    def _create_environment(self, df: pd.DataFrame) -> TradingEnvironment:
        """Create a trading environment."""
        return TradingEnvironment(
            df=df,
            features=self.features,
            window_size=self.window_size,
            transaction_cost=self.transaction_cost,
            reward_scaling=self.reward_scaling,
        )

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None, **kwargs) -> "LSTMPPO":
        """Fit the LSTM-PPO model to training data using NNX."""
        try:
            self.feature_names = X_train.columns.tolist()
            train_df = X_train.copy()
            if y_train is not None and self.target_column not in train_df.columns:
                train_df[self.target_column] = y_train
            elif self.target_column not in train_df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in training data.")

            self.env = self._create_environment(train_df)

            self.trainer = PPOTrainer(
                env=self.env,
                network_kwargs=self.network_kwargs,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                seed=self.seed,
                device=self.device,
            )

            total_timesteps = kwargs.get("total_timesteps", 100000)

            self.model_state, self.graph_def = self.trainer.learn(total_timesteps)
            self.rngs = self.trainer.rngs

            self.is_fitted = True
            logger.info("LSTM-PPO NNX model trained")

        except Exception as e:
            logger.error(f"Error training LSTM-PPO NNX model: {e}", exc_info=True)
            self.is_fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (actions) using the fitted LSTM-PPO NNX model."""
        if not self.is_fitted or self.model_state is None or self.graph_def is None:
            logger.warning("LSTM-PPO NNX model not fitted or state/graph_def missing")
            return np.array([])

        try:
            test_df = X.copy()
            if self.target_column not in test_df.columns:
                logger.warning(
                    f"Target column '{self.target_column}' missing in prediction data. Adding dummy column."
                )
                test_df[self.target_column] = 0.0

            test_env = self._create_environment(test_df)
            obs_np, _ = test_env.reset()
            obs: jnp.ndarray = jnp.array(obs_np)
            predictions = []

            graph_def_static = self.graph_def
            model_state_static = self.model_state

            @partial(jax.jit, static_argnames=("graph_def",))
            def predict_step(graph_def, current_state, observations, rngs):
                # Use nnx.merge instead of instance method
                model = nnx.merge(graph_def, current_state)
                # Create a new RNG
                model_rngs = nnx.Rngs(params=0)
                pi, _ = model(observations, rngs=model_rngs)
                action = pi.mode()
                # Return original RNGs
                return action, rngs

            pred_rngs = self.rngs if self.rngs else nnx.Rngs(sample=self.seed + 100)

            done = False
            steps = 0
            max_steps = len(X)

            while not done and steps < max_steps:
                obs_batch = obs[None, :]
                action, pred_rngs_next = predict_step(graph_def_static, model_state_static, obs_batch, pred_rngs)
                # Update RNG state for next prediction step
                pred_rngs = pred_rngs_next.fork()
                action_np = np.array(action.squeeze())
                predictions.append(float(action_np))

                next_obs, _, terminated, truncated, _ = test_env.step(action_np)
                done = terminated or truncated
                obs = jnp.array(next_obs)
                steps += 1

            if len(predictions) < len(X):
                padding = np.zeros(len(X) - len(predictions))
                predictions.extend(padding)

            return np.array(predictions[: len(X)])

        except Exception as e:
            logger.error(f"Error predicting with LSTM-PPO NNX model: {e}", exc_info=True)
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save the NNX model state and graph definition."""
        directory = directory / "lstm_ppo_nnx"
        directory.mkdir(exist_ok=True, parents=True)
        model_dir = directory / f"{self.name}_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)
        state_path = model_dir / "model.nnx.state"
        graph_path = model_dir / "model.nnx.graphdef"
        config_path = model_dir / "config.pkl"

        if self.is_fitted and self.model_state is not None and self.graph_def is not None:
            try:
                # Use nnx state save/load API if available, else pickle state
                # Checking documentation/issues, direct save/load might not be stable yet.
                # Fallback to pickling state for now.
                # nnx.save_state(state_path, self.model_state)
                with open(state_path, "wb") as f:
                    pickle.dump(self.model_state, f)
                with open(graph_path, "wb") as f:
                    pickle.dump(self.graph_def, f)

                config = {
                    "target_column": self.target_column,
                    "horizon": self.horizon,
                    "features": self.features,
                    "window_size": self.window_size,
                    "network_kwargs": self.network_kwargs,
                    "learning_rate": self.learning_rate,
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_range,
                    "transaction_cost": self.transaction_cost,
                    "reward_scaling": self.reward_scaling,
                    "is_fitted": self.is_fitted,
                    "feature_names": self.feature_names,
                    "seed": self.seed,
                }
                with open(config_path, "wb") as f:
                    pickle.dump(config, f)
                logger.info(f"LSTM-PPO NNX model saved to {model_dir}")
            except Exception as e:
                logger.error(f"Error saving LSTM-PPO NNX model: {e}", exc_info=True)
        else:
            logger.warning("LSTM-PPO NNX model not saved: model is not fitted or state/graph_def is None.")

        return model_dir

    def load(self, model_path: Path) -> "LSTMPPO":
        """Load NNX model state and graph definition from disk."""
        try:
            state_path = model_path / "model.nnx.state"
            graph_path = model_path / "model.nnx.graphdef"
            config_path = model_path / "config.pkl"

            if not state_path.exists() or not graph_path.exists() or not config_path.exists():
                logger.error(f"Cannot load LSTM-PPO NNX model: Missing files in {model_path}")
                self.is_fitted = False
                return self

            with open(config_path, "rb") as f:
                config = pickle.load(f)

            self.target_column = config["target_column"]
            self.horizon = config["horizon"]
            self.features = config["features"]
            self.window_size = config["window_size"]
            self.network_kwargs = config["network_kwargs"]
            self.learning_rate = config["learning_rate"]
            self.n_steps = config["n_steps"]
            self.batch_size = config["batch_size"]
            self.n_epochs = config["n_epochs"]
            self.gamma = config["gamma"]
            self.gae_lambda = config["gae_lambda"]
            self.clip_range = config["clip_range"]
            self.transaction_cost = config["transaction_cost"]
            self.reward_scaling = config["reward_scaling"]
            self.feature_names = config.get("feature_names", [])
            self.seed = config.get("seed", 42)
            self.is_fitted = config["is_fitted"]

            if self.is_fitted:
                # Use pickle to load state as nnx.load_state might not be stable/available
                # self.model_state = nnx.load_state(state_path)
                with open(state_path, "rb") as f:
                    self.model_state = pickle.load(f)
                with open(graph_path, "rb") as f:
                    self.graph_def = pickle.load(f)

                self.rngs = nnx.Rngs(params=self.seed, sample=self.seed + 1)
                dummy_df_len = self.window_size + 1
                dummy_features = self.features if self.features else [self.target_column]
                dummy_df = pd.DataFrame({feature: np.zeros(dummy_df_len) for feature in dummy_features})
                if self.target_column not in dummy_df.columns:
                    dummy_df[self.target_column] = 0.0
                self.env = self._create_environment(dummy_df)
                self.trainer = None

                logger.info(f"LSTM-PPO NNX model loaded from {model_path}")
            else:
                logger.warning(f"Model loaded from {model_path}, but was not marked as fitted.")
                self.model_state = None
                self.graph_def = None

        except Exception as e:
            logger.error(f"Error loading LSTM-PPO NNX model from {model_path}: {e}", exc_info=True)
            self.is_fitted = False
            self.model_state = None
            self.graph_def = None

        return self
