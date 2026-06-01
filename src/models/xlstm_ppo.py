# src/models/xlstm_ppo.py
"""xLSTM-PPO implementation using JAX/Flax NNX for reinforcement learning trading.

The xLSTM cell (sLSTMBlock) implements the stabilizer + normalizer variant from
Beck et al. 2024 (https://arxiv.org/abs/2405.04517). Here it is used as a
window-feature-extractor inside an Actor-Critic network: the LSTM is scanned
across the observation window each call, with carry reset between calls. This
mirrors the design of the LSTM-PPO model in `lstm_ppo.py` and lets us share
its tested PPO training loop verbatim.
"""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import distrax  # type: ignore
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pandas as pd
from gymnasium import spaces

from src.config import MODELS_DIR
from src.models.base import BaseModel
from src.models.lstm_ppo import TradingEnvironment

logger = logging.getLogger(__name__)


class sLSTMBlock(nnx.Module):
    """sLSTM cell (xLSTM paper): standard LSTM gates with stabilizer + normalizer."""

    def __init__(self, *, hidden_dim: int, input_features: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        din = input_features + hidden_dim
        self.input_proj = nnx.Linear(in_features=din, out_features=hidden_dim, rngs=rngs)
        self.forget_proj = nnx.Linear(in_features=din, out_features=hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(in_features=din, out_features=hidden_dim, rngs=rngs)
        self.cell_proj = nnx.Linear(in_features=din, out_features=hidden_dim, rngs=rngs)

    def __call__(self, carry, x):
        """One step. carry = (h, c, n), x of shape (batch, input_features)."""
        h_prev, c_prev, n_prev = carry
        xh = jnp.concatenate([x, h_prev], axis=-1)
        i_gate = nnx.sigmoid(self.input_proj(xh))
        f_gate = nnx.sigmoid(self.forget_proj(xh))
        o_gate = nnx.sigmoid(self.output_proj(xh))
        z_tilde = nnx.tanh(self.cell_proj(xh))
        n_curr = n_prev * f_gate + i_gate
        c_curr = (c_prev * f_gate + z_tilde * i_gate) / (n_curr + 1e-6)
        h_curr = o_gate * nnx.tanh(c_curr)
        return (h_curr, c_curr, n_curr), h_curr

    @staticmethod
    def initialize_carry(batch_size: int, hidden_dim: int):
        h = jnp.zeros((batch_size, hidden_dim))
        c = jnp.zeros((batch_size, hidden_dim))
        n = jnp.ones((batch_size, hidden_dim))
        return (h, c, n)


class XLSTMFeatureExtractor(nnx.Module):
    """Scans an sLSTM cell over the time dimension of a window observation."""

    def __init__(
        self,
        *,
        input_features: int,
        features_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.cell = sLSTMBlock(hidden_dim=hidden_dim, input_features=input_features, rngs=rngs)
        self.dense = nnx.Linear(in_features=hidden_dim, out_features=features_dim, rngs=rngs)

    def __call__(self, observations: jax.Array) -> jax.Array:
        """observations: (batch, window, features) → (batch, features_dim)."""
        if observations.ndim == 2:
            observations = observations[:, None, :]
        batch_size = observations.shape[0]
        carry = sLSTMBlock.initialize_carry(batch_size, self.hidden_dim)
        # Time-major scan.
        obs_t = jnp.transpose(observations, (1, 0, 2))
        for i in range(obs_t.shape[0]):
            carry, _ = self.cell(carry, obs_t[i])
        h_last = carry[0]
        return nnx.relu(self.dense(h_last))


class ActorCriticXLSTM(nnx.Module):
    """Actor-Critic head over the xLSTM feature extractor."""

    def __init__(
        self,
        *,
        input_features: int,
        action_dim: int,
        features_dim: int = 64,
        hidden_dim: int = 128,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.feature_extractor = XLSTMFeatureExtractor(
            input_features=input_features,
            features_dim=features_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.actor_mean = nnx.Linear(in_features=features_dim, out_features=action_dim, rngs=rngs)
        self.actor_logstd = nnx.Param(jnp.zeros(action_dim))
        self.critic_value = nnx.Linear(in_features=features_dim, out_features=1, rngs=rngs)

    def __call__(self, observations: jax.Array) -> Tuple[distrax.Distribution, jax.Array]:
        features = self.feature_extractor(observations)
        mean = self.actor_mean(features)
        std = jnp.exp(self.actor_logstd.value)
        pi = distrax.MultivariateNormalDiag(mean, std)
        value = jnp.squeeze(self.critic_value(features), axis=-1)
        return pi, value


class XLSTMPPOTrainer:
    """PPO trainer for the xLSTM Actor-Critic. Mirrors `PPOTrainer` in lstm_ppo."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
    optimizer_state: Optional[optax.OptState] = None
    graph_def: Optional[nnx.GraphDef] = None

    def __init__(
        self,
        env: TradingEnvironment,
        network_kwargs: Dict[str, Any],
        learning_rate: float = 3e-4,
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

        self._key = jax.random.key(seed)
        self.rngs = nnx.Rngs(params=seed, sample=seed + 1, carry=seed + 2, default=seed + 3)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        self._initialize_model()

    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    def _initialize_model(self):
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise ValueError("Only Box spaces are supported")

        input_features = obs_space.shape[-1]
        action_dim = action_space.shape[0]
        network = ActorCriticXLSTM(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )
        graph_def, params, rngs_state = nnx.split(network, nnx.Param, ...)
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
        model = nnx.merge(graph_def, params, rngs_state)
        pi, value = model(observations)
        actions = pi.sample(seed=key)
        log_probs = pi.log_prob(actions)
        _, _new_params, new_rngs_state = nnx.split(model, nnx.Param, ...)
        return actions, log_probs, value, new_rngs_state

    def _get_action_and_value(self, observations: np.ndarray):
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer state not initialized")
        actions, log_probs, value, new_rngs = self._get_action_and_value_jit(
            self.graph_def, self.params, self.rngs_state, jnp.asarray(observations), self._next_key()
        )
        self.rngs_state = new_rngs
        return np.asarray(actions), np.asarray(log_probs), np.asarray(value)

    def _compute_gae(self, rewards: jax.Array, values: jax.Array, dones: jax.Array, next_value: jax.Array):
        n = len(rewards)
        advantages = jnp.zeros_like(rewards)
        last = jnp.array(0.0, dtype=jnp.float32)
        v_ext = jnp.append(values, next_value)
        for t in reversed(range(n)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * v_ext[t + 1] * nonterm - v_ext[t]
            advantages = advantages.at[t].set(delta + self.gamma * self.gae_lambda * nonterm * last)
            last = advantages[t]
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
        vloss = 0.5 * jnp.maximum((values - batch["returns"]) ** 2, (v_clipped - batch["returns"]) ** 2).mean()

        total = policy_loss + vf_coef * vloss - ent_coef * entropy
        info = {
            "policy_loss": policy_loss,
            "value_loss": vloss,
            "entropy": entropy,
            "approx_kl": 0.5 * jnp.mean((log_probs - batch["log_probs"]) ** 2),
            "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32)),
        }
        return total, info

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
        def loss_fn(p: nnx.State):
            return XLSTMPPOTrainer._ppo_loss(graph_def, p, rngs_state, batch, clip_range, vf_coef, ent_coef)

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, info

    def _update_policy(self, rollout: Dict[str, jnp.ndarray]):
        if self.params is None or self.optimizer_state is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer not initialized")
        n = rollout["observations"].shape[0]
        idx = jnp.arange(n)
        last_info: Dict[str, jnp.ndarray] = {}
        for _ in range(self.n_epochs):
            idx = jax.random.permutation(self._next_key(), idx)
            for start in range(0, n, self.batch_size):
                b_idx = idx[start : start + self.batch_size]
                batch = {k: v[b_idx] for k, v in rollout.items()}
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
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer not initialized")

        logger.info(f"JAX devices: {jax.devices()}, backend: {jax.default_backend()}")
        obs_np, _ = self.env.reset()
        obs = jnp.asarray(obs_np)
        n_updates = max(1, total_timesteps // self.n_steps)

        from tqdm import tqdm

        for update in tqdm(range(n_updates), desc="Training xLSTM-PPO"):
            obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
            for _ in range(self.n_steps):
                a_np, lp_np, v_np = self._get_action_and_value(obs[None, :])
                a_scalar = np.asarray(a_np).reshape(-1)
                obs_buf.append(obs)
                act_buf.append(jnp.asarray(a_scalar))
                val_buf.append(float(np.asarray(v_np).squeeze()))
                logp_buf.append(float(np.asarray(lp_np).squeeze()))
                next_obs, r, term, trunc, _ = self.env.step(a_scalar)
                rew_buf.append(float(r))
                done_buf.append(float(term or trunc))
                if term or trunc:
                    next_obs, _ = self.env.reset()
                obs = jnp.asarray(next_obs)

            _, _, last_v = self._get_action_and_value(obs[None, :])
            rewards = jnp.asarray(rew_buf, dtype=jnp.float32)
            values = jnp.asarray(val_buf, dtype=jnp.float32)
            dones = jnp.asarray(done_buf, dtype=jnp.float32)
            returns, adv = self._compute_gae(rewards, values, dones, jnp.asarray(last_v).squeeze())

            rollout = {
                "observations": jnp.stack(obs_buf),
                "actions": jnp.stack(act_buf),
                "log_probs": jnp.asarray(logp_buf, dtype=jnp.float32),
                "values": values,
                "returns": returns,
                "advantages": adv,
            }
            self.params, self.optimizer_state, info = self._update_policy(rollout)
            if update % 10 == 0 and info:
                logger.info(
                    f"Update {update}/{n_updates} | "
                    f"policy_loss={float(info['policy_loss']):.4f} "
                    f"value_loss={float(info['value_loss']):.4f} "
                    f"entropy={float(info['entropy']):.4f} "
                    f"approx_kl={float(info['approx_kl']):.4f} "
                    f"clip_frac={float(info['clip_frac']):.3f}"
                )

        logger.info(f"Completed {n_updates} xLSTM-PPO updates")
        return self.params, self.rngs_state, self.graph_def


class XLSTMPPOAgent(BaseModel):
    """BaseModel-compatible wrapper around `XLSTMPPOTrainer`."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
    graph_def: Optional[nnx.GraphDef] = None
    trainer: Optional[XLSTMPPOTrainer] = None
    env: Optional[TradingEnvironment] = None

    def __init__(self, name: str = "xlstm_ppo", target_column: str = "close", horizon: int = 5, **kwargs):
        super().__init__(name=name, target_column=target_column, horizon=horizon, **kwargs)
        self.features: List[str] = kwargs.get("features") or ["open", "high", "low", "close", "volume"]
        self.window_size: int = kwargs.get("window_size", 20)
        self.network_kwargs: Dict[str, Any] = kwargs.get(
            "network_kwargs", {"features_dim": 64, "hidden_dim": 128}
        )
        self.learning_rate: float = kwargs.get("learning_rate", 3e-4)
        self.n_steps: int = kwargs.get("n_steps", 2048)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.n_epochs: int = kwargs.get("n_epochs", 10)
        self.gamma: float = kwargs.get("gamma", 0.99)
        self.gae_lambda: float = kwargs.get("gae_lambda", 0.95)
        self.clip_range: float = kwargs.get("clip_range", 0.2)
        self.transaction_cost: float = kwargs.get("transaction_cost", 0.001)
        self.reward_scaling: float = kwargs.get("reward_scaling", 1.0)
        self.seed: int = kwargs.get("seed", 42)

        self.params = None
        self.rngs_state = None
        self.graph_def = None
        self.trainer = None
        self.env = None
        self.is_fitted = False
        self.feature_names = []

    @property
    def required_history(self) -> int:
        """predict() builds a TradingEnvironment from X, which asserts
        len(df) >= window_size + 2."""
        return self.window_size + 2

    def _create_environment(self, df: pd.DataFrame) -> TradingEnvironment:
        return TradingEnvironment(
            df=df,
            features=self.features,
            window_size=self.window_size,
            transaction_cost=self.transaction_cost,
            reward_scaling=self.reward_scaling,
        )

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None, **kwargs) -> "XLSTMPPOAgent":
        self.feature_names = X_train.columns.tolist()
        train_df = X_train.copy()
        if y_train is not None and self.target_column not in train_df.columns:
            train_df[self.target_column] = y_train
        elif self.target_column not in train_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not in training data")

        self.env = self._create_environment(train_df)
        self.trainer = XLSTMPPOTrainer(
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
        total_timesteps = kwargs.get("total_timesteps", 100_000)
        self.params, self.rngs_state, self.graph_def = self.trainer.learn(total_timesteps)
        self.is_fitted = True
        logger.info("xLSTM-PPO model trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.params is None or self.graph_def is None or self.rngs_state is None:
            logger.warning("xLSTM-PPO not fitted; returning empty array")
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
        return np.asarray(predictions[: len(X)], dtype=np.float32)

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Persist params, rngs_state, and config. graph_def is rebuilt at
        load time (nnx.GraphDef can't be pickled — flax initializers carry
        local-closure refs)."""
        save_dir = directory / "xlstm_ppo" / f"{self.name}_h{self.horizon}"
        save_dir.mkdir(exist_ok=True, parents=True)
        if not (self.is_fitted and self.params is not None and self.rngs_state is not None):
            logger.warning("xLSTM-PPO not saved: model is not fitted")
            return save_dir

        with open(save_dir / "params.pkl", "wb") as f:
            pickle.dump(self.params, f)
        with open(save_dir / "rngs_state.pkl", "wb") as f:
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
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        logger.info(f"xLSTM-PPO saved to {save_dir}")
        return save_dir

    def load(self, model_path: Path) -> "XLSTMPPOAgent":
        cfg = model_path / "config.pkl"
        params_p = model_path / "params.pkl"
        rngs_p = model_path / "rngs_state.pkl"
        if not all(p.exists() for p in (cfg, params_p, rngs_p)):
            logger.error(f"Cannot load xLSTM-PPO: missing files in {model_path}")
            self.is_fitted = False
            return self
        with open(cfg, "rb") as f:
            config = pickle.load(f)
        for k in (
            "target_column", "horizon", "features", "window_size", "network_kwargs",
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
            "clip_range", "transaction_cost", "reward_scaling", "seed",
        ):
            setattr(self, k, config[k])
        self.feature_names = config.get("feature_names", [])
        self.is_fitted = config["is_fitted"]
        if not self.is_fitted:
            return self

        with open(params_p, "rb") as f:
            self.params = pickle.load(f)
        with open(rngs_p, "rb") as f:
            self.rngs_state = pickle.load(f)
        self.rngs = nnx.Rngs(
            params=self.seed, sample=self.seed + 1, carry=self.seed + 2, default=self.seed + 3
        )
        input_features = config.get("input_features", len(self.features))
        action_dim = config.get("action_dim", 1)
        network = ActorCriticXLSTM(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )
        self.graph_def, _, _ = nnx.split(network, nnx.Param, ...)
        logger.info(f"xLSTM-PPO loaded from {model_path}")
        return self
