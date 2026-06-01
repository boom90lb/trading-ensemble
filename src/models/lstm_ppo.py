# src/models/lstm_ppo.py
"""LSTM-PPO model implementation using JAX/Flax for reinforcement learning trading."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        # Create the LSTM cell
        self.lstm_cell = nnx.LSTMCell(in_features=input_features, hidden_features=lstm_hidden_size, rngs=rngs)

        # Create the output projection layer
        self.dense = nnx.Linear(in_features=lstm_hidden_size, out_features=features_dim, rngs=rngs)

    def __call__(self, observations: jax.Array) -> jax.Array:
        """Forward pass through the feature extractor."""
        # Handle 2D input (add sequence dimension)
        if observations.ndim == 2:
            observations = observations[:, None, :]

        batch_size, seq_len, _ = observations.shape

        # Initialize the LSTM carry state
        init_carry = self.lstm_cell.initialize_carry((batch_size,))

        # Transpose to time-major format for processing
        # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        observations_t = jnp.transpose(observations, (1, 0, 2))

        # Process each timestep
        carry = init_carry
        hidden_states = []

        for i in range(observations_t.shape[0]):
            carry, hidden = self.lstm_cell(carry, observations_t[i])
            hidden_states.append(hidden)

        # Stack hidden states
        hidden_states_array = jnp.stack(hidden_states)

        # Use the last hidden state
        last_hidden = hidden_states_array[-1] if seq_len > 0 else hidden_states_array[0]

        # Project to feature dimension
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

    def __call__(self, observations: jax.Array, **kwargs) -> Tuple[distrax.Distribution, jax.Array]:
        """Forward pass through the Actor-Critic network.

        Args:
            observations: Input observations
            **kwargs: Additional keyword arguments (including rngs) that are ignored

        Returns:
            Tuple of policy distribution and value estimate
        """
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
            transaction_cost: Transaction cost ratio (per unit position change)
            initial_balance: Initial account balance
            reward_scaling: Scaling factor for rewards
        """
        super().__init__()

        if not 0.0 <= transaction_cost < 0.1:
            raise ValueError(f"transaction_cost must be in [0, 0.1), got {transaction_cost}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if len(df) < window_size + 2:
            raise ValueError(f"DataFrame too short ({len(df)}) for window_size={window_size}")

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
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        self.current_step: int = 0
        self.current_position: float = 0.0
        self.current_balance: float = float(initial_balance)
        self.done: bool = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        num_features = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, num_features), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        if options and "start_idx" in options:
            start_idx = options["start_idx"]
        else:
            start_idx = self.window_size

        # Must leave room for at least one (t, t+1) reward step.
        self.current_step = max(self.window_size, min(start_idx, len(self.df) - 2))
        self.current_position = 0.0
        self.current_balance = float(self.initial_balance)
        self.done = False

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment.

        Reward formulation (López-de-Prado-style, per-step):
            r_t = position_t * (close_{t+1} - close_t) / close_t
                  - transaction_cost * |position_t - position_{t-1}|

        The action sets `position_t`; the realized return is the next-bar move,
        net of round-trip-proportional turnover cost on the position change.
        """
        if self.done:
            logger.warning("step() on a done environment; returning terminal observation.")
            return self._get_observation(), 0.0, True, False, self._get_info()

        if not 0 <= self.current_step < len(self.df) - 1:
            raise RuntimeError(
                f"Invalid current_step={self.current_step} for len(df)={len(self.df)}; "
                "step requires t+1 to be a valid index."
            )

        new_position = float(np.clip(action.item() if hasattr(action, "item") else action, -1.0, 1.0))

        price_t = float(self.df.iloc[self.current_step]["close"])
        price_t1 = float(self.df.iloc[self.current_step + 1]["close"])
        if price_t <= 0:
            raise RuntimeError(f"Non-positive close price at step {self.current_step}: {price_t}")

        bar_return = (price_t1 - price_t) / price_t
        turnover = abs(new_position - self.current_position)
        cost = turnover * self.transaction_cost
        reward = (new_position * bar_return - cost) * self.reward_scaling

        # Mark-to-market the notional balance for diagnostics (not used in reward).
        self.current_balance *= 1.0 + new_position * bar_return - cost
        self.current_position = new_position
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        if terminated:
            self.done = True

        return self._get_observation(), float(reward), bool(terminated), truncated, self._get_info()

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
        """Diagnostics; not part of the reward."""
        idx = min(self.current_step, len(self.df) - 1)
        return {
            "balance": self.current_balance,
            "position": self.current_position,
            "step": self.current_step,
            "price": float(self.df.iloc[idx]["close"]),
        }


class PPOTrainer:
    """PPO trainer implementation using JAX/Flax NNX."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
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
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        seed: int = 42,
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
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        # JAX PRNG key (consumed/split as we go; never reset to a constant)
        self._key = jax.random.key(seed)

        self.rngs = nnx.Rngs(params=seed, sample=seed + 1, carry=seed + 2, default=seed + 3)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        self._initialize_model()

    def _next_key(self) -> jax.Array:
        """Return a fresh PRNG subkey, advancing internal state."""
        self._key, sub = jax.random.split(self._key)
        return sub

    def _initialize_model(self):
        """Build the ActorCritic and split into (graph_def, params, rngs_state)."""
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise ValueError("Currently only supports Box observation and action spaces")

        input_features = obs_space.shape[-1]
        action_dim = action_space.shape[0]

        network_instance = ActorCritic(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )

        # Split params (gradient-bearing) from rngs/counter state (non-grad).
        graph_def, params, rngs_state = nnx.split(network_instance, nnx.Param, ...)
        self.graph_def = graph_def
        self.params = params
        self.rngs_state = rngs_state
        self.optimizer_state = self.optimizer.init(params)

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def",))
    def _get_action_and_value_jit(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        observations: jax.Array,
        key: jax.Array,
    ):
        """Sample an action and value from the current policy."""
        model = nnx.merge(graph_def, params, rngs_state)
        pi, value = model(observations)
        actions = pi.sample(seed=key)
        log_probs = pi.log_prob(actions)
        # Forward may advance internal rng counters; capture the new non-Param state.
        _, _new_params, new_rngs_state = nnx.split(model, nnx.Param, ...)
        return actions, log_probs, value, new_rngs_state

    def _get_action_and_value(self, observations):
        """Numpy-facing wrapper around the JIT'd action sampler."""
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Model state/graph_def not initialized.")

        observations_jax = jnp.asarray(observations)
        actions, log_probs, value, new_rngs_state = self._get_action_and_value_jit(
            self.graph_def, self.params, self.rngs_state, observations_jax, self._next_key()
        )
        self.rngs_state = new_rngs_state
        return np.asarray(actions), np.asarray(log_probs), np.asarray(value)

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute generalized advantage estimation."""
        n_steps = len(rewards)
        advantages = jnp.zeros_like(rewards)
        last_gae_lam = jnp.array(0.0, dtype=jnp.float32)
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
    @partial(jax.jit, static_argnames=("graph_def", "clip_range", "vf_coef", "ent_coef"))
    def _ppo_loss(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        batch: Dict,
        clip_range: float,
        vf_coef: float,
        ent_coef: float,
    ):
        """Clipped-objective PPO loss (Schulman et al. 2017), value-clipped."""
        model = nnx.merge(graph_def, params, rngs_state)
        pi, values = model(batch["observations"])
        log_probs = pi.log_prob(batch["actions"])
        entropy = pi.entropy().mean()

        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio = jnp.exp(log_probs - batch["log_probs"])
        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        v_clipped = batch["values"] + jnp.clip(values - batch["values"], -clip_range, clip_range)
        vloss1 = (values - batch["returns"]) ** 2
        vloss2 = (v_clipped - batch["returns"]) ** 2
        value_loss = 0.5 * jnp.maximum(vloss1, vloss2).mean()

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": 0.5 * jnp.mean((log_probs - batch["log_probs"]) ** 2),
            "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32)),
        }
        return total_loss, info

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "clip_range", "vf_coef", "ent_coef", "optimizer"))
    def _update_step_jit(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        optimizer_state: optax.OptState,
        batch: Dict,
        clip_range: float,
        vf_coef: float,
        ent_coef: float,
        optimizer: optax.GradientTransformation,
    ):
        """One PPO gradient step. Differentiates loss wrt params only."""

        def loss_fn(p: nnx.State):
            return PPOTrainer._ppo_loss(graph_def, p, rngs_state, batch, clip_range, vf_coef, ent_coef)

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_optimizer_state, loss, info

    def _update_policy(self, rollout_data: Dict[str, jnp.ndarray]):
        """Run n_epochs of minibatch SGD over the rollout."""
        if self.params is None or self.optimizer_state is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer state components not initialized before update.")

        n_samples = rollout_data["observations"].shape[0]
        indices = jnp.arange(n_samples)
        last_info: Dict[str, jnp.ndarray] = {}

        for _ in range(self.n_epochs):
            indices = jax.random.permutation(self._next_key(), indices)
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                batch = {k: v[batch_indices] for k, v in rollout_data.items()}

                self.params, self.optimizer_state, _, last_info = self._update_step_jit(
                    self.graph_def,
                    self.params,
                    self.rngs_state,
                    self.optimizer_state,
                    batch,
                    self.clip_range,
                    self.vf_coef,
                    self.ent_coef,
                    self.optimizer,
                )

        return self.params, self.optimizer_state, last_info

    def learn(self, total_timesteps: int) -> Tuple[nnx.State, nnx.State, nnx.GraphDef]:
        """Train the PPO agent. Returns (params, rngs_state, graph_def)."""
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer state not initialized before learning.")

        logger.info(f"JAX devices: {jax.devices()}, backend: {jax.default_backend()}")

        obs_np, _ = self.env.reset()
        obs = jnp.asarray(obs_np)

        n_updates = max(1, total_timesteps // self.n_steps)
        from tqdm import tqdm

        for update in tqdm(range(n_updates), desc="Training LSTM-PPO"):
            obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

            for _ in range(self.n_steps):
                action_np, log_prob_np, value_np = self._get_action_and_value(obs[None, :])
                action_scalar = np.asarray(action_np).reshape(-1)

                obs_buf.append(obs)
                act_buf.append(jnp.asarray(action_scalar))
                val_buf.append(float(np.asarray(value_np).squeeze()))
                logp_buf.append(float(np.asarray(log_prob_np).squeeze()))

                next_obs, reward, terminated, truncated, _ = self.env.step(action_scalar)
                done = bool(terminated or truncated)

                rew_buf.append(float(reward))
                done_buf.append(float(done))

                if done:
                    obs_np, _ = self.env.reset()
                    obs = jnp.asarray(obs_np)
                else:
                    obs = jnp.asarray(next_obs)

            # Bootstrap value for the last state and compute GAE.
            _, _, last_value_np = self._get_action_and_value(obs[None, :])
            rewards = jnp.asarray(rew_buf, dtype=jnp.float32)
            values = jnp.asarray(val_buf, dtype=jnp.float32)
            dones = jnp.asarray(done_buf, dtype=jnp.float32)
            returns, advantages = self._compute_gae(
                rewards, values, dones, jnp.asarray(last_value_np).squeeze()
            )

            update_data = {
                "observations": jnp.stack(obs_buf),
                "actions": jnp.stack(act_buf),
                "log_probs": jnp.asarray(logp_buf, dtype=jnp.float32),
                "values": values,
                "returns": returns,
                "advantages": advantages,
            }

            self.params, self.optimizer_state, info = self._update_policy(update_data)

            if update % 10 == 0 and info:
                logger.info(
                    f"Update {update}/{n_updates} | "
                    f"policy_loss={float(info['policy_loss']):.4f} "
                    f"value_loss={float(info['value_loss']):.4f} "
                    f"entropy={float(info['entropy']):.4f} "
                    f"approx_kl={float(info['approx_kl']):.4f} "
                    f"clip_frac={float(info['clip_frac']):.3f}"
                )

        logger.info(f"Completed {n_updates} PPO updates")
        return self.params, self.rngs_state, self.graph_def


class LSTMPPO(BaseModel):
    """LSTM-PPO reinforcement learning model for trading using JAX/Flax NNX."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
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
        self.params = None
        self.rngs_state = None
        self.graph_def = None
        self.trainer = None
        self.rngs = None

    @property
    def required_history(self) -> int:
        """predict() builds a TradingEnvironment from X, which asserts
        len(df) >= window_size + 2 (needs the window plus one (t, t+1) step)."""
        return self.window_size + 2

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
        """Fit the LSTM-PPO model. Raises on failure rather than silently un-fitting."""
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
        )

        total_timesteps = kwargs.get("total_timesteps", 100000)
        self.params, self.rngs_state, self.graph_def = self.trainer.learn(total_timesteps)
        self.rngs = self.trainer.rngs

        self.is_fitted = True
        logger.info("LSTM-PPO NNX model trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Roll out the learned policy deterministically (mode action) over X."""
        if not self.is_fitted or self.params is None or self.graph_def is None or self.rngs_state is None:
            logger.warning("LSTM-PPO not fitted; returning empty prediction array")
            return np.array([])

        test_df = X.copy()
        if self.target_column not in test_df.columns:
            test_df[self.target_column] = 0.0

        test_env = self._create_environment(test_df)
        obs_np, _ = test_env.reset()
        obs = jnp.asarray(obs_np)
        predictions: List[float] = []

        graph_def = self.graph_def
        rngs_state = self.rngs_state

        @partial(jax.jit, static_argnames=("graph_def",))
        def predict_step(graph_def, params, rngs_state, observations):
            model = nnx.merge(graph_def, params, rngs_state)
            pi, _ = model(observations)
            return pi.mode()

        done = False
        while not done and len(predictions) < len(X):
            action = predict_step(graph_def, self.params, rngs_state, obs[None, :])
            action_np = np.asarray(action).reshape(-1)
            predictions.append(float(action_np.item() if action_np.size == 1 else action_np[0]))
            next_obs, _, terminated, truncated, _ = test_env.step(action_np)
            done = bool(terminated or truncated)
            obs = jnp.asarray(next_obs)

        if len(predictions) < len(X):
            predictions.extend([0.0] * (len(X) - len(predictions)))

        logger.info(f"Generated {len(predictions)} LSTM-PPO predictions")
        return np.asarray(predictions[: len(X)], dtype=np.float32)

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Persist params, rngs_state, and config. graph_def is reconstructed
        at load time — nnx.GraphDef pickling fails on closure references in
        flax initializers (e.g. variance_scaling.<locals>.init)."""
        directory = directory / "lstm_ppo_nnx"
        directory.mkdir(exist_ok=True, parents=True)
        model_dir = directory / f"{self.name}_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)

        if not (self.is_fitted and self.params is not None and self.rngs_state is not None):
            logger.warning("LSTM-PPO not saved: model is not fitted")
            return model_dir

        with open(model_dir / "params.pkl", "wb") as f:
            pickle.dump(self.params, f)
        with open(model_dir / "rngs_state.pkl", "wb") as f:
            pickle.dump(self.rngs_state, f)

        config = {
            "target_column": self.target_column,
            "horizon": self.horizon,
            "features": self.features,
            "input_features": len(self.features),
            "action_dim": 1,
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
        with open(model_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        logger.info(f"LSTM-PPO saved to {model_dir}")
        return model_dir

    def load(self, model_path: Path) -> "LSTMPPO":
        """Load params, rngs_state, and config. graph_def is rebuilt by
        instantiating a fresh ActorCritic with the saved input_features /
        action_dim / network_kwargs and re-splitting."""
        config_path = model_path / "config.pkl"
        params_path = model_path / "params.pkl"
        rngs_path = model_path / "rngs_state.pkl"

        if not all(p.exists() for p in (config_path, params_path, rngs_path)):
            logger.error(f"Cannot load LSTM-PPO: missing files in {model_path}")
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

        if not self.is_fitted:
            logger.warning(f"Model file present at {model_path} but not marked fitted")
            self.params = None
            self.rngs_state = None
            self.graph_def = None
            return self

        with open(params_path, "rb") as f:
            self.params = pickle.load(f)
        with open(rngs_path, "rb") as f:
            self.rngs_state = pickle.load(f)

        self.rngs = nnx.Rngs(
            params=self.seed, sample=self.seed + 1, carry=self.seed + 2, default=self.seed + 3
        )
        input_features = config.get("input_features", len(self.features))
        action_dim = config.get("action_dim", 1)
        network = ActorCritic(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )
        self.graph_def, _, _ = nnx.split(network, nnx.Param, ...)

        logger.info(f"LSTM-PPO loaded from {model_path}")
        return self
