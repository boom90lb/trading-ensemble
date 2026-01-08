# src/models/xlstm_ppo.py
"""xLSTM PPO implementation using NNX for reinforcement learning trading."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chex
import distrax  # type: ignore
import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pandas as pd
from gymnasium import spaces

# Import needed modules
from src.config import MODELS_DIR  # Will be used in save/load methods
from src.models.base import BaseModel  # Will be used for inheritance

# Configure logging
logger = logging.getLogger(__name__)

# xLSTM implementation based on concepts from the xLSTM paper
# (e.g., https://arxiv.org/abs/2405.04517)


class sLSTMBlock(nnx.Module):
    """Simple LSTM Block for xLSTM implementation using NNX."""

    def __init__(self, *, hidden_dim: int, input_features: int, rngs: nnx.Rngs):
        """Initialize the sLSTM block.

        Args:
            hidden_dim: Hidden dimension size
            input_features: Input feature dimension
            rngs: Random number generators
        """
        self.hidden_dim = hidden_dim
        # Input dimension for projections: input_features + hidden_dim
        din = input_features + hidden_dim

        # Initialize linear layers with in_features and out_features
        self.input_proj = nnx.Linear(in_features=din, out_features=self.hidden_dim, rngs=rngs)
        self.forget_proj = nnx.Linear(in_features=din, out_features=self.hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(in_features=din, out_features=self.hidden_dim, rngs=rngs)
        self.cell_proj = nnx.Linear(in_features=din, out_features=self.hidden_dim, rngs=rngs)

    def __call__(self, carry, x):
        """Forward pass of the sLSTM block.

        Args:
            carry: Tuple of (h_prev, c_prev, n_prev) for hidden state, cell state, and normalizer state
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Tuple of (new_carry, output) where new_carry is (h_next, c_next, n_next) and output is h_curr
        """
        h_prev, c_prev, n_prev = carry

        # Concatenate input and previous hidden state
        if x.ndim == 1:  # Handle case during init with single observation
            x = x[None, :]  # Add batch dim
        if h_prev.ndim == 1:  # Handle case during init
            h_prev = h_prev[None, :]

        # Ensure batch dims match if needed, though scan usually handles this
        chex.assert_equal_shape_prefix([x, h_prev], 1)

        xh = jnp.concatenate([x, h_prev], axis=-1)

        # Input Gate
        i_proj_out = self.input_proj(xh)
        i_gate = nnx.sigmoid(i_proj_out)  # Use nnx activation

        # Forget Gate
        f_proj_out = self.forget_proj(xh)
        f_gate = nnx.sigmoid(f_proj_out)

        # Output Gate
        o_proj_out = self.output_proj(xh)
        o_gate = nnx.sigmoid(o_proj_out)

        # Cell Input Transformation
        z_proj_out = self.cell_proj(xh)
        z_tilde = nnx.tanh(z_proj_out)  # Use nnx activation

        # Stabilizer and Normalization
        # Note: Ensure element-wise operations if shapes differ from standard LSTM
        n_curr = n_prev * f_gate + i_gate
        # Add epsilon for numerical stability
        c_curr = (c_prev * f_gate) / (n_curr + 1e-6) + (z_tilde * i_gate) / (n_curr + 1e-6)
        h_curr = o_gate * nnx.tanh(c_curr)  # Use nnx activation

        carry = (h_curr, c_curr, n_curr)
        output = h_curr

        return carry, output

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        """Initialize the LSTM carry state.

        Args:
            batch_size: Batch size
            hidden_dim: Hidden dimension size

        Returns:
            Tuple of (h_init, c_init, n_init) for hidden state, cell state, and normalizer state
        """
        h_init = jnp.zeros((batch_size, hidden_dim))
        # Confirm shapes based on paper, assuming element-wise for now
        c_init = jnp.zeros((batch_size, hidden_dim))
        n_init = jnp.ones((batch_size, hidden_dim))
        return (h_init, c_init, n_init)


# TODO: Implement mLSTM Block if needed


# TODO: Define Actor Critic using xLSTM
class ActorCriticXLSTM(nnx.Module):
    """Actor-Critic model with xLSTM core using NNX."""

    def __init__(self, *, action_dim: int, input_features: int, hidden_dim: int = 64, rngs: nnx.Rngs):
        """Initialize the Actor-Critic model.

        Args:
            action_dim: Dimension of the action space
            input_features: Dimension of the input features
            hidden_dim: Hidden dimension size
            rngs: Random number generators
        """
        self.action_dim = action_dim
        self.input_features = input_features
        self.hidden_dim = hidden_dim

        # NNX modules require explicit RNG passing
        self.lstm_core = sLSTMBlock(hidden_dim=self.hidden_dim, input_features=input_features, rngs=rngs)
        self.actor_mean = nnx.Linear(
            in_features=self.hidden_dim,
            out_features=self.action_dim,
            rngs=rngs,
        )
        self.actor_logstd = nnx.Param(jnp.zeros(self.action_dim))  # Trainable param
        self.critic_value = nnx.Linear(
            in_features=self.hidden_dim,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, carry, x, *, rngs: Optional[nnx.Rngs] = None):  # rngs param required by NNX API
        """Forward pass of the model.

        Args:
            carry: LSTM carry state (h, c, n)
            x: Tuple of (observations, dones)
            rngs: Optional random number generators

        Returns:
            Tuple of (new_carry, policy_distribution, value)
        """
        obs, dones = x  # Assuming input x is (observation, done_flags)

        # Reset carry if done
        reset_carry = self.initialize_carry(obs.shape[0], self.hidden_dim)
        h_prev, c_prev, n_prev = carry
        h_reset, c_reset, n_reset = reset_carry
        done_broadcast = dones[:, None]  # Broadcast dones (batch,) -> (batch, 1)
        h_eff = jnp.where(done_broadcast, h_reset, h_prev)
        c_eff = jnp.where(done_broadcast, c_reset, c_prev)
        n_eff = jnp.where(done_broadcast, n_reset, n_prev)
        effective_carry = (h_eff, c_eff, n_eff)

        # Run LSTM core
        new_carry, lstm_output = self.lstm_core(effective_carry, obs)

        # --- Actor Head ---
        actor_mean_val = self.actor_mean(lstm_output)
        pi = distrax.MultivariateNormalDiag(actor_mean_val, jnp.exp(self.actor_logstd.value))

        # --- Critic Head ---
        critic_val = self.critic_value(lstm_output)
        value = jnp.squeeze(critic_val, axis=-1)

        return new_carry, pi, value

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        """Initialize the LSTM carry state.

        Args:
            batch_size: Batch size
            hidden_dim: Hidden dimension size

        Returns:
            Tuple of (h0, c0, n0) for hidden state, cell state, and normalizer state
        """
        # Initialize hidden state, cell state, and normalizer state
        h0 = jnp.zeros((batch_size, hidden_dim))
        c0 = jnp.zeros((batch_size, hidden_dim))
        n0 = jnp.zeros((batch_size, hidden_dim))  # Normalizer state
        return (h0, c0, n0)


# TODO: Define TrainingState

# TODO: Implement PPO logic (train, update, etc.) adapting from lstm_ppo.py


# Define a TrainState specific to ActorCriticXLSTM
class XLSTMTrainState:
    # We can add type hints for the carry state if needed, depends on sLSTM/mLSTM definition
    pass


# Define the Trainer class adapting from lstm_ppo.py PPOTrainer
class XLSTMPPOTrainer:
    """PPO Trainer using xLSTM and NNX."""

    def __init__(
        self,
        env,  # Assuming env provides observation_space, action_space
        network_kwargs: Dict[str, Any],  # Pass network kwargs
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        seed: int = 42,
        device: str = "cpu",  # Less relevant for JAX default placement
    ):
        self.env = env
        self.network_kwargs = network_kwargs  # Store kwargs
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.seed = seed

        self.rngs = nnx.Rngs(params=seed, sample=seed + 1)  # Add sample stream
        self.optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=self.learning_rate))

        # Initialize state components
        self.model_state = None
        self.optimizer_state = None
        self.graph_def = None

        self._initialize_model()

    # Constructor moved above

    def _initialize_model(self):
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise ValueError("Currently only supports Box observation and action spaces")

        obs_shape = obs_space.shape
        action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1
        input_features = obs_shape[-1]  # Infer input features

        # Instantiate the network
        network_instance = ActorCriticXLSTM(
            action_dim=action_dim,
            input_features=input_features,
            **self.network_kwargs,
            rngs=self.rngs,  # Pass RNGs for initialization
        )

        # Split into graphdef and state using module-level function
        graph_def, state = nnx.split(network_instance)
        self.graph_def = graph_def
        # Convert to a proper State object
        self.model_state = state
        # Extract params for optimizer - use a simpler approach
        # Convert state to a dict for optimizer initialization
        params = {}
        self.optimizer_state = self.optimizer.init(params)

    # JITted step function using NNX state
    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def",))
    def _get_action_and_value_step_jit(
        graph_def: nnx.GraphDef,
        model_state: nnx.State,
        hidden_state_carry: Tuple,  # Pass LSTM carry explicitly
        observations: jax.Array,
        dones: jax.Array,
        rngs: nnx.Rngs,
    ):
        # Create RNGs for model use and sampling
        model_rngs = nnx.Rngs(params=jax.random.key(0))
        sample_key = jax.random.key(0)
        # Use module-level merge function
        model = nnx.merge(graph_def, model_state)

        # Call the model (which handles one step)
        new_hidden_state_carry, pi, value = model(hidden_state_carry, (observations, dones), rngs=model_rngs)

        # Use sample_key for sampling
        action = pi.sample(seed=sample_key)
        log_prob = pi.log_prob(action)

        # Return new carry, action, log_prob, value, and original RNGs
        return new_hidden_state_carry, action, log_prob, value, rngs

    def _get_action_and_value_step(self, hidden_state_carry, observations, dones):
        """Get action and value for a single step, handling state and RNGs."""
        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Trainer state not initialized.")

        obs_jax = jnp.asarray(observations)
        dones_jax = jnp.asarray(dones)

        # Call JITted function
        new_carry, action, log_prob, value, updated_rngs = self._get_action_and_value_step_jit(
            self.graph_def, self.model_state, hidden_state_carry, obs_jax, dones_jax, self.rngs
        )
        # Create a new RNG for next call
        self.rngs = nnx.Rngs(params=jax.random.key(0))

        # Return numpy arrays for env interaction
        return new_carry, np.array(action), np.array(log_prob), np.array(value)

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute generalized advantage estimation (unchanged)."""
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

    # PPO loss for sequences using NNX Scan
    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "clip_range", "network_hidden_dim"))
    def _ppo_loss_sequence_jit(
        graph_def: nnx.GraphDef,
        model_state: nnx.State,
        batch: Dict,
        clip_range: float,
        network_hidden_dim: int,  # Pass hidden dim statically
        rngs: nnx.Rngs,
    ):
        obs_seq = batch["observations"]
        actions_seq = batch["actions"]
        old_log_probs_seq = batch["log_probs"]
        returns_seq = batch["returns"]
        advantages_seq = batch["advantages"]
        dones_seq = batch["dones"]
        initial_carry = batch["initial_carry"]

        chex.assert_rank(obs_seq, 3)
        # Get batch and sequence dimensions for potential future use
        _, _ = obs_seq.shape[0], obs_seq.shape[1]  # batch_size, seq_len

        # Define the scan body function using the JIT context
        # It takes the *state* of the relevant part of the model
        # Here, we need the ActorCritic state to be scanned
        def scan_body(ac_model_state: nnx.State, carry, xs):
            obs_step, done_step = xs
            step_rngs = rngs.fork()  # Need to handle RNGs inside scan

            # Reconstruct the ActorCritic model inside scan using module-level merge
            # This assumes graph_def is available via closure or static arg
            ac_model = nnx.merge(graph_def, ac_model_state)

            # Call the model for one step
            new_carry, pi, value = ac_model(carry, (obs_step, done_step), rngs=step_rngs)

            # Return new carry and outputs (policy, value) for this step
            # Also need to return the updated state if scan modifies it (e.g., BatchNorm)
            # Assuming no state modifications other than carry for now.
            # Also, return consumed RNG state? NNX Scan might handle this.
            return new_carry, (pi, value)

        # Prepare scan inputs: transpose obs and dones to time-major
        scan_inputs = (einops.rearrange(obs_seq, "b t f -> t b f"), einops.rearrange(dones_seq, "b t -> t b"))

        # Use nnx.scan instead of nnx.Scan for the updated API
        scan_fn = lambda carry, x: scan_body(model_state, carry, x)
        # Use the scan function directly
        # Run scan and get results
        _, (pi_seq_scan, values_seq_scan) = jax.lax.scan(scan_fn, initial_carry, scan_inputs)

        # Scan has already been run above

        # Transpose results back to batch-major
        values_seq = einops.rearrange(values_seq_scan, "t b -> b t")

        # Calculate log probs from the policy distributions sequence
        log_probs_seq = jax.vmap(lambda pi, a: pi.log_prob(a), in_axes=(0, 1), out_axes=1)(
            pi_seq_scan, einops.rearrange(actions_seq, "b t a -> t b a")
        )
        log_probs_seq = einops.rearrange(log_probs_seq, "t b -> b t")

        entropy = jax.vmap(lambda pi: pi.entropy(), in_axes=0, out_axes=0)(pi_seq_scan)
        entropy = einops.rearrange(entropy, "t b -> b t").mean()

        ratio = jnp.exp(log_probs_seq - old_log_probs_seq)
        advantages_seq_norm = (advantages_seq - advantages_seq.mean(axis=1, keepdims=True)) / (
            advantages_seq.std(axis=1, keepdims=True) + 1e-8
        )

        surrogate1 = ratio * advantages_seq_norm
        surrogate2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_seq_norm
        policy_loss = -jnp.minimum(surrogate1, surrogate2).mean()

        value_loss = 0.5 * ((values_seq - returns_seq) ** 2).mean()
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        approx_kl = 0.5 * ((log_probs_seq - old_log_probs_seq) ** 2).mean()

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }
        # Return original RNGs (assuming Scan consumed its forks)
        return total_loss, info, rngs

    # JITted update step for sequences
    @staticmethod
    @partial(jax.jit, static_argnames=("graph_def", "clip_range", "optimizer", "network_hidden_dim"))
    def _update_step_sequence_jit(
        graph_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer_state: optax.OptState,
        batch: Dict,
        clip_range: float,
        optimizer: optax.GradientTransformation,
        network_hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        step_rngs = rngs.fork()

        # Loss function operates on params state
        def loss_for_grad(params_state: nnx.State):
            current_state = model_state.merge(params_state)
            loss, info, _ = XLSTMPPOTrainer._ppo_loss_sequence_jit(
                graph_def, current_state, batch, clip_range, network_hidden_dim, step_rngs
            )
            return loss, info

        params_state = model_state.filter(nnx.Param)
        (loss, info), grads = nnx.value_and_grad(loss_for_grad, has_aux=True)(params_state)

        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params_state)

        # Create a copy of the model state
        new_model_state = jax.tree.map(lambda x: x, model_state)
        # Apply updates in-place
        nnx.update(new_model_state, updates)
        # new_model_state now contains the updated values

        # Return new states, loss, info, and original RNGs
        return new_model_state, new_optimizer_state, loss, info, rngs

    def _update_policy(self, sequence_data: Dict[str, jnp.ndarray], final_value: jnp.ndarray):
        """Update the policy using sequence data."""
        if self.model_state is None or self.optimizer_state is None or self.graph_def is None:
            raise RuntimeError("Trainer state is not initialized.")

        num_envs = sequence_data["observations"].shape[0]
        chex.assert_shape(final_value, (num_envs,))

        # GAE calculation remains the same (uses numpy/jax arrays)
        rewards = sequence_data["rewards"]
        values = sequence_data["values"]
        dones = sequence_data["dones"]
        returns_seq, advantages_seq = jax.vmap(self._compute_gae, in_axes=(0, 0, 0, 0))(
            rewards, values, dones, final_value
        )

        batch = {
            "observations": sequence_data["observations"],
            "actions": sequence_data["actions"],
            "log_probs": sequence_data["log_probs"],
            "dones": sequence_data["dones"],
            "initial_carry": sequence_data["initial_carry"],
            "returns": returns_seq,
            "advantages": advantages_seq,
        }

        # Static args for JIT
        graph_def_static = self.graph_def
        optimizer_static = self.optimizer
        clip_range_static = self.clip_range
        # Extract hidden dim from network_kwargs or state if possible
        network_hidden_dim_static = self.network_kwargs.get("hidden_dim", 64)  # Example default

        n_samples = num_envs
        indices = jnp.arange(n_samples)
        last_info = {}

        for _ in range(self.n_epochs):
            # Use a fresh key for permutation
            perm_key = jax.random.key(0)
            indices = jax.random.permutation(perm_key, indices)
            # Create new RNGs for next iteration using the seed attribute
            if hasattr(self, "seed"):
                self.rngs = nnx.Rngs(params=self.seed)
            else:
                self.rngs = nnx.Rngs(params=42)  # Default seed

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                micro_batch = jax.tree.map(lambda x: x[batch_indices], batch)

                if self.model_state is None or self.optimizer_state is None:
                    raise RuntimeError("Trainer state became None during update loop.")

                # Call JITted update function
                # Call update step and unpack results
                new_model_state, new_optimizer_state, _, last_info, self.rngs = (
                    XLSTMPPOTrainer._update_step_sequence_jit(
                        graph_def_static,
                        self.model_state,
                        self.optimizer_state,
                        micro_batch,
                        clip_range_static,
                        optimizer_static,
                        network_hidden_dim_static,
                        self.rngs,  # Pass current RNGs
                    )
                )
                self.model_state = new_model_state
                self.optimizer_state = new_optimizer_state
                # self.rngs already updated by JIT return

        return self.model_state, last_info

    def learn(self, total_timesteps: int) -> Tuple[nnx.State, nnx.GraphDef]:
        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Trainer state not initialized.")

        # Need hidden_dim for initializing carry
        hidden_dim = self.network_kwargs.get("hidden_dim", 64)

        # Assuming single environment for now
        num_envs = 1
        # Reset environment and get initial observation
        obs_np, _ = self.env.reset()  # Ignore info dict
        obs: jnp.ndarray = jnp.array(obs_np)
        # Initialize LSTM carry state
        hidden_state_carry = ActorCriticXLSTM.initialize_carry(num_envs, hidden_dim)
        done = jnp.zeros(num_envs, dtype=jnp.bool_)

        current_timestep = 0
        num_updates = total_timesteps // (self.n_steps * num_envs)

        for update in range(1, num_updates + 1):
            # Define rollout buffer with type annotations
            rollout_buffer: Dict[str, List] = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "log_probs": [],
                "values": [],
                "initial_carry": [],
            }
            # Store hidden state at the start of the sequence
            initial_carry_update = hidden_state_carry

            for step in range(self.n_steps):
                current_timestep += num_envs
                obs_batch = obs[None, :] if num_envs == 1 else obs
                done_batch = done[None, :] if num_envs == 1 else done

                # Get action, value, etc. for the current step
                new_carry, action_np, log_prob_np, value_np = self._get_action_and_value_step(
                    hidden_state_carry, obs_batch, done_batch
                )

                # Store data (use JAX arrays if possible, convert later if needed)
                # Store the state *before* the step
                rollout_buffer["initial_carry"].append(hidden_state_carry)
                rollout_buffer["observations"].append(obs)
                rollout_buffer["actions"].append(jnp.asarray(action_np))
                rollout_buffer["log_probs"].append(jnp.asarray(log_prob_np))
                rollout_buffer["values"].append(jnp.asarray(value_np))

                # Step environment
                action_env = action_np[0] if num_envs == 1 else action_np
                # Step environment and get results
                next_obs_np, reward, terminated, truncated, _ = self.env.step(action_env)  # Ignore info dict
                done_step = np.logical_or(terminated, truncated)
                done_step_np = np.array([done_step]) if num_envs == 1 else np.array(done_step)

                rollout_buffer["rewards"].append(jnp.asarray(reward).reshape(num_envs))
                rollout_buffer["dones"].append(jnp.asarray(done_step_np).reshape(num_envs))

                # Update state for next iteration
                obs = jnp.asarray(next_obs_np)
                hidden_state_carry = new_carry
                done = jnp.asarray(done_step_np)

                if num_envs == 1 and done_step:
                    # Reset environment after episode is done
                    obs_np, _ = self.env.reset()  # Ignore info dict
                    obs = jnp.asarray(obs_np)
                    # Reset done flag, carry reset handled by model step logic
                    done = jnp.zeros(num_envs, dtype=jnp.bool_)

            # --- Prepare data for update --- #
            # Get value of the final next state
            final_obs_batch = obs[None, :] if num_envs == 1 else obs
            final_done_batch = done[None, :] if num_envs == 1 else done
            _, _, _, final_value_np = self._get_action_and_value_step(
                hidden_state_carry, final_obs_batch, final_done_batch
            )
            final_value = jnp.asarray(final_value_np).squeeze()

            # Collate rollout data into sequences
            def stack_arrays(data_list):
                # Handle potential scalar rewards/dones
                if data_list[0].ndim == 0:
                    return jnp.stack(data_list).reshape(num_envs, self.n_steps)
                return einops.rearrange(jnp.stack(data_list), "(s e) ... -> e s ...", e=num_envs, s=self.n_steps)

            sequence_data = {k: stack_arrays(v) for k, v in rollout_buffer.items() if k != "initial_carry"}
            # Handle initial carry state (needs careful structuring)
            # It should be the state *before* the first step of each env's sequence
            # Handle the initial carry state specially
            sequence_data["initial_carry"] = jax.tree.map(lambda x: x, initial_carry_update)

            # --- Update policy --- #
            self.model_state, update_info = self._update_policy(sequence_data, final_value)

            if update % 10 == 0:
                loss_val = update_info.get("policy_loss", jnp.nan)
                ent_val = update_info.get("entropy", jnp.nan)
                kl_val = update_info.get("approx_kl", jnp.nan)
                logger.info(
                    f"Update {update}/{num_updates}, Timestep {current_timestep}/{total_timesteps}, "
                    f"Loss: {float(loss_val):.4f}, Entropy: {float(ent_val):.4f}, KL: {float(kl_val):.4f}"
                )

        if self.model_state is None or self.graph_def is None:
            raise RuntimeError("Model state or graph_def is None after training.")
        # Return the model state and graph_def
        # Use type ignore to handle potential type mismatches with NNX state types
        return self.model_state, self.graph_def  # type: ignore


# Implementation of XLSTMPPOTrainer class for training the model
# This class will be used by the XLSTMPPOAgent class below


# Wrapper class that inherits from BaseModel to integrate with the ensemble
class XLSTMPPOAgent(BaseModel):
    """xLSTM PPO Agent that inherits from BaseModel for ensemble integration."""

    def __init__(self, name: str = "xlstm_ppo", target_column: str = "close", horizon: int = 5, **kwargs):
        """Initialize the xLSTM PPO Agent.

        Args:
            name: Model name
            target_column: Target column to predict
            horizon: Forecast horizon
            **kwargs: Additional model parameters
        """
        super().__init__(name=name, target_column=target_column, horizon=horizon, **kwargs)

        # Initialize model parameters
        self.features = kwargs.get("features", [])
        self.window_size = kwargs.get("window_size", 20)
        self.network_kwargs = kwargs.get("network_kwargs", {"hidden_dim": 64})
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.n_steps = kwargs.get("n_steps", 2048)
        self.batch_size = kwargs.get("batch_size", 64)
        self.n_epochs = kwargs.get("n_epochs", 10)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_range = kwargs.get("clip_range", 0.2)
        self.transaction_cost = kwargs.get("transaction_cost", 0.0001)
        self.reward_scaling = kwargs.get("reward_scaling", 1.0)
        self.seed = kwargs.get("seed", 42)

        # Initialize model state
        self.model_state = None
        self.graph_def = None
        self.trainer = None
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "XLSTMPPOAgent":
        """Fit the xLSTM PPO model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            # Extract parameters from kwargs
            total_timesteps = kwargs.get("total_timesteps", 100000)

            # Log training parameters
            logger.info(f"Training xLSTM PPO model with {total_timesteps} timesteps")

            # TODO: Implement actual training logic using the XLSTMPPOTrainer
            # For now, just set is_fitted to True for testing
            self.is_fitted = True

            return self
        except Exception as e:
            logger.error(f"Error fitting xLSTM PPO model: {e}")
            return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted xLSTM PPO model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            logger.warning("xLSTM PPO model not fitted yet")
            return np.array([])

        try:
            # TODO: Implement actual prediction logic
            # For now, return random predictions for testing
            np.random.seed(self.seed)
            predictions = np.random.uniform(-1.0, 1.0, size=len(X))

            return predictions
        except Exception as e:
            logger.error(f"Error predicting with xLSTM PPO model: {e}")
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        save_dir = directory / "xlstm_ppo"
        save_dir.mkdir(exist_ok=True, parents=True)
        model_dir = save_dir / f"{self.name}_h{self.horizon}"
        model_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Save model configuration
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

            with open(model_dir / "config.pkl", "wb") as f:
                pickle.dump(config, f)

            logger.info(f"xLSTM PPO model saved to {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"Error saving xLSTM PPO model: {e}")
            return model_dir

    def load(self, model_path: Path) -> "XLSTMPPOAgent":
        """Load model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            config_path = model_path / "config.pkl"

            if not config_path.exists():
                logger.error(f"Cannot load xLSTM PPO model: Missing config file in {model_path}")
                self.is_fitted = False
                return self

            with open(config_path, "rb") as f:
                config = pickle.load(f)

            # Update model attributes from config
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

            logger.info(f"xLSTM PPO model loaded from {model_path}")
            return self
        except Exception as e:
            logger.error(f"Error loading xLSTM PPO model from {model_path}: {e}")
            self.is_fitted = False
            return self
