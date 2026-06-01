# src/models/xlstm_grpo.py
"""xLSTM-GRPO implementation using JAX/Flax NNX.

Group Relative Policy Optimization (DeepSeekMath, 2024) adapted to a continuous
trading control. For each "prompt" (a start state in `TradingEnvironment`) we
sample G action sequences from the current policy, score each by cumulative
discounted reward, and apply a pairwise DPO-style preference loss between every
ranked (preferred, dispreferred) pair. No critic, no reference model — the
implicit baseline is the within-group mean (the "relative" in GRPO).

This file deliberately mirrors the structure of `xlstm_ppo.py` (window-feature
xLSTM extractor + Gaussian policy head) so the two RL agents are
interoperable in the ensemble.
"""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

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
from src.models.xlstm_ppo import XLSTMFeatureExtractor

logger = logging.getLogger(__name__)


class PolicyXLSTM(nnx.Module):
    """Gaussian policy over actions, conditioned on a window observation."""

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

    def __call__(self, observations: jax.Array) -> distrax.Distribution:
        features = self.feature_extractor(observations)
        mean = self.actor_mean(features)
        std = jnp.exp(self.actor_logstd.value)
        return distrax.MultivariateNormalDiag(mean, std)


class XLSTMGRPOTrainer:
    """GRPO trainer: G rollouts per prompt, ranked by cumulative discounted reward,
    pairwise DPO loss with within-group mean as implicit baseline."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
    optimizer_state: Optional[optax.OptState] = None
    graph_def: Optional[nnx.GraphDef] = None

    def __init__(
        self,
        env: TradingEnvironment,
        network_kwargs: Dict[str, Any],
        learning_rate: float = 5e-5,
        beta: float = 0.1,
        group_size: int = 4,
        prompts_per_update: int = 4,
        sequence_length: int = 16,
        gamma: float = 0.99,
        n_epochs: int = 1,
        max_grad_norm: float = 0.5,
        seed: int = 42,
    ):
        self.env = env
        self.network_kwargs = network_kwargs
        self.learning_rate = learning_rate
        self.beta = beta
        self.group_size = group_size
        self.prompts_per_update = prompts_per_update
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        if group_size < 2:
            raise ValueError("group_size must be >= 2 for pairwise preference loss")

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
        network = PolicyXLSTM(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )
        graph_def, params, rngs_state = nnx.split(network, nnx.Param, ...)
        self.graph_def = graph_def
        self.params = params
        self.rngs_state = rngs_state
        self.optimizer_state = self.optimizer.init(params)

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def",))
    def _sample_action_jit(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        observation: jax.Array,
        key: jax.Array,
    ):
        model = nnx.merge(graph_def, params, rngs_state)
        pi = model(observation)
        action = pi.sample(seed=key)
        _, _new_params, new_rngs_state = nnx.split(model, nnx.Param, ...)
        return action, new_rngs_state

    def _sample_action(self, observation: np.ndarray) -> np.ndarray:
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer not initialized")
        action, new_rngs = self._sample_action_jit(
            self.graph_def, self.params, self.rngs_state, jnp.asarray(observation), self._next_key()
        )
        self.rngs_state = new_rngs
        return np.asarray(action)

    def _rollout_one(
        self, start_idx: int, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roll out the current policy for `sequence_length` steps starting at `start_idx`.

        Returns: (observations, actions, rewards) with shapes
            obs:     (T, window, features)
            actions: (T, action_dim)
            rewards: (T,)
        """
        obs_np, _ = self.env.reset(options={"start_idx": start_idx})
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        rew_list: List[float] = []
        for _ in range(sequence_length):
            obs_list.append(obs_np.copy())
            action = self._sample_action(obs_np[None, :]).reshape(-1)
            next_obs, r, terminated, truncated, _ = self.env.step(action)
            act_list.append(np.asarray(action))
            rew_list.append(float(r))
            if terminated or truncated:
                break
            obs_np = next_obs
        T = len(obs_list)
        # Pad to sequence_length if early termination so all G rollouts share a length.
        if T < sequence_length:
            pad_obs = np.zeros_like(obs_list[0])
            pad_act = np.zeros_like(act_list[0])
            for _ in range(sequence_length - T):
                obs_list.append(pad_obs)
                act_list.append(pad_act)
                rew_list.append(0.0)
        return (
            np.stack(obs_list).astype(np.float32),
            np.stack(act_list).astype(np.float32),
            np.asarray(rew_list, dtype=np.float32),
        )

    def _generate_group_sequences(
        self, start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate G rollouts from the same starting state.

        Returns batched (G, T, ...) arrays.
        """
        obs_g: List[np.ndarray] = []
        act_g: List[np.ndarray] = []
        rew_g: List[np.ndarray] = []
        for _ in range(self.group_size):
            o, a, r = self._rollout_one(start_idx, self.sequence_length)
            obs_g.append(o)
            act_g.append(a)
            rew_g.append(r)
        return np.stack(obs_g), np.stack(act_g), np.stack(rew_g)

    def _rank_sequences(self, rewards: np.ndarray) -> np.ndarray:
        """Rank rollouts by cumulative discounted reward, best-first."""
        T = rewards.shape[-1]
        discount = self.gamma ** np.arange(T, dtype=np.float32)
        cum = (rewards * discount).sum(axis=-1)  # (G,)
        return np.argsort(-cum)  # descending

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def",))
    def _sequence_log_prob(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        observations: jax.Array,  # (T, window, features)
        actions: jax.Array,  # (T, action_dim)
    ) -> jax.Array:
        """Sum of per-step log probs under the current policy."""
        model = nnx.merge(graph_def, params, rngs_state)
        pi = model(observations)
        log_probs = pi.log_prob(actions)  # (T,)
        return jnp.sum(log_probs)

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "beta"))
    def _grpo_loss(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        group_obs: jax.Array,  # (B, G, T, window, features)
        group_actions: jax.Array,  # (B, G, T, action_dim)
        rank_perm: jax.Array,  # (B, G) — rank_perm[b, r] = group-index of rank-r rollout
        beta: float,
    ):
        # Per-rollout sequence log prob, shape (B, G).
        def per_group_logp(obs_g, act_g):
            return jax.vmap(
                lambda o, a: XLSTMGRPOTrainer._sequence_log_prob(graph_def, params, rngs_state, o, a)
            )(obs_g, act_g)

        group_logp = jax.vmap(per_group_logp)(group_obs, group_actions)  # (B, G)

        # Reorder to rank-major: ranked_logp[b, r] = log prob of the r-th best rollout in prompt b.
        ranked_logp = jnp.take_along_axis(group_logp, rank_perm, axis=1)  # (B, G)

        # Pairwise preference loss: every (i < j) means rank-i preferred over rank-j.
        G = ranked_logp.shape[1]
        diffs = ranked_logp[:, :, None] - ranked_logp[:, None, :]  # (B, G, G)
        pair_loss = -jax.nn.log_sigmoid(beta * diffs)  # (B, G, G)
        # Upper-triangular mask: i < j.
        mask = jnp.triu(jnp.ones((G, G), dtype=jnp.float32), k=1)
        loss = (pair_loss * mask).sum(axis=(1, 2)) / mask.sum()
        loss = loss.mean()

        info = {
            "grpo_loss": loss,
            "avg_logp": jnp.mean(group_logp),
            "best_logp": jnp.mean(ranked_logp[:, 0]),
            "worst_logp": jnp.mean(ranked_logp[:, -1]),
            "logp_spread": jnp.mean(ranked_logp[:, 0] - ranked_logp[:, -1]),
        }
        return loss, info

    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "beta", "optimizer"))
    def _update_step_jit(
        graph_def: nnx.GraphDef,
        params: nnx.State,
        rngs_state: nnx.State,
        optimizer_state: optax.OptState,
        group_obs: jax.Array,
        group_actions: jax.Array,
        rank_perm: jax.Array,
        beta: float,
        optimizer: optax.GradientTransformation,
    ):
        def loss_fn(p: nnx.State):
            return XLSTMGRPOTrainer._grpo_loss(
                graph_def, p, rngs_state, group_obs, group_actions, rank_perm, beta
            )

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, info

    def train(self, total_updates: int) -> Tuple[nnx.State, nnx.State, nnx.GraphDef]:
        if self.params is None or self.graph_def is None or self.rngs_state is None:
            raise RuntimeError("Trainer not initialized")

        logger.info(f"JAX devices: {jax.devices()}, backend: {jax.default_backend()}")
        from tqdm import tqdm

        n_bars = len(self.env.df)
        min_start = self.env.window_size
        # Leave room for sequence_length steps from any chosen start.
        max_start = max(min_start + 1, n_bars - self.sequence_length - 1)

        for update in tqdm(range(total_updates), desc="Training xLSTM-GRPO"):
            # Sample B prompts (start indices), generate G rollouts each, rank.
            group_obs_list: List[np.ndarray] = []
            group_act_list: List[np.ndarray] = []
            rank_perm_list: List[np.ndarray] = []
            for _ in range(self.prompts_per_update):
                start_key = self._next_key()
                start_idx = int(jax.random.randint(start_key, (), min_start, max_start))
                obs_g, act_g, rew_g = self._generate_group_sequences(start_idx)
                rank_perm_list.append(self._rank_sequences(rew_g).astype(np.int32))
                group_obs_list.append(obs_g)
                group_act_list.append(act_g)

            group_obs = jnp.asarray(np.stack(group_obs_list))  # (B,G,T,W,F)
            group_actions = jnp.asarray(np.stack(group_act_list))  # (B,G,T,A)
            rank_perm = jnp.asarray(np.stack(rank_perm_list))  # (B,G)

            for _ in range(self.n_epochs):
                self.params, self.optimizer_state, loss, info = self._update_step_jit(
                    self.graph_def,
                    self.params,
                    self.rngs_state,
                    self.optimizer_state,
                    group_obs,
                    group_actions,
                    rank_perm,
                    self.beta,
                    self.optimizer,
                )

            if update % 10 == 0:
                logger.info(
                    f"Update {update}/{total_updates} | "
                    f"loss={float(loss):.4f} avg_logp={float(info['avg_logp']):.4f} "
                    f"best={float(info['best_logp']):.4f} worst={float(info['worst_logp']):.4f} "
                    f"spread={float(info['logp_spread']):.4f}"
                )

        logger.info(f"Completed {total_updates} GRPO updates")
        return self.params, self.rngs_state, self.graph_def


class XLSTMGRPOAgent(BaseModel):
    """BaseModel-compatible wrapper around `XLSTMGRPOTrainer`."""

    params: Optional[nnx.State] = None
    rngs_state: Optional[nnx.State] = None
    graph_def: Optional[nnx.GraphDef] = None
    trainer: Optional[XLSTMGRPOTrainer] = None
    env: Optional[TradingEnvironment] = None

    def __init__(self, name: str = "xlstm_grpo", target_column: str = "close", horizon: int = 5, **kwargs):
        super().__init__(name=name, target_column=target_column, horizon=horizon, **kwargs)
        self.features: List[str] = kwargs.get("features") or ["open", "high", "low", "close", "volume"]
        self.window_size: int = kwargs.get("window_size", 20)
        self.network_kwargs: Dict[str, Any] = kwargs.get(
            "network_kwargs", {"features_dim": 64, "hidden_dim": 128}
        )
        self.learning_rate: float = kwargs.get("learning_rate", 5e-5)
        self.beta: float = kwargs.get("beta", 0.1)
        self.group_size: int = kwargs.get("group_size", 4)
        self.prompts_per_update: int = kwargs.get("prompts_per_update", 4)
        self.sequence_length: int = kwargs.get("sequence_length", 16)
        self.gamma: float = kwargs.get("gamma", 0.99)
        self.n_epochs: int = kwargs.get("n_epochs", 1)
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

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None, **kwargs) -> "XLSTMGRPOAgent":
        self.feature_names = X_train.columns.tolist()
        train_df = X_train.copy()
        if y_train is not None and self.target_column not in train_df.columns:
            train_df[self.target_column] = y_train
        elif self.target_column not in train_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not in training data")

        self.env = self._create_environment(train_df)
        self.trainer = XLSTMGRPOTrainer(
            env=self.env,
            network_kwargs=self.network_kwargs,
            learning_rate=self.learning_rate,
            beta=self.beta,
            group_size=self.group_size,
            prompts_per_update=self.prompts_per_update,
            sequence_length=self.sequence_length,
            gamma=self.gamma,
            n_epochs=self.n_epochs,
            seed=self.seed,
        )
        total_updates = kwargs.get("total_updates", 1000)
        self.params, self.rngs_state, self.graph_def = self.trainer.train(total_updates)
        self.is_fitted = True
        logger.info("xLSTM-GRPO model trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.params is None or self.graph_def is None or self.rngs_state is None:
            logger.warning("xLSTM-GRPO not fitted; returning empty array")
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
            pi = model(observations)
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
        save_dir = directory / "xlstm_grpo" / f"{self.name}_h{self.horizon}"
        save_dir.mkdir(exist_ok=True, parents=True)
        if not (self.is_fitted and self.params is not None and self.rngs_state is not None):
            logger.warning("xLSTM-GRPO not saved: model is not fitted")
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
            "beta": self.beta,
            "group_size": self.group_size,
            "prompts_per_update": self.prompts_per_update,
            "sequence_length": self.sequence_length,
            "gamma": self.gamma,
            "n_epochs": self.n_epochs,
            "transaction_cost": self.transaction_cost,
            "reward_scaling": self.reward_scaling,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "seed": self.seed,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        logger.info(f"xLSTM-GRPO saved to {save_dir}")
        return save_dir

    def load(self, model_path: Path) -> "XLSTMGRPOAgent":
        cfg = model_path / "config.pkl"
        params_p = model_path / "params.pkl"
        rngs_p = model_path / "rngs_state.pkl"
        if not all(p.exists() for p in (cfg, params_p, rngs_p)):
            logger.error(f"Cannot load xLSTM-GRPO: missing files in {model_path}")
            self.is_fitted = False
            return self
        with open(cfg, "rb") as f:
            config = pickle.load(f)
        for k in (
            "target_column", "horizon", "features", "window_size", "network_kwargs",
            "learning_rate", "beta", "group_size", "prompts_per_update", "sequence_length",
            "gamma", "n_epochs", "transaction_cost", "reward_scaling", "seed",
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
        network = PolicyXLSTM(
            input_features=input_features, action_dim=action_dim, **self.network_kwargs, rngs=self.rngs
        )
        self.graph_def, _, _ = nnx.split(network, nnx.Param, ...)
        logger.info(f"xLSTM-GRPO loaded from {model_path}")
        return self
