# src/models/xlstm_grpo.py
"""xLSTM GRPO implementation using NNX for reinforcement learning trading."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# xLSTM Core components using NNX
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
        c_init = jnp.zeros((batch_size, hidden_dim))
        n_init = jnp.ones((batch_size, hidden_dim))  # Normalizer initialized to ones
        return (h_init, c_init, n_init)


# Network definition using NNX
class PolicyNetworkXLSTM(nnx.Module):
    """Policy network with xLSTM core using NNX for GRPO."""

    def __init__(self, *, action_dim: int, input_features: int, hidden_dim: int = 64, rngs: nnx.Rngs):
        """Initialize the policy network.

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

    def __call__(self, carry, x):
        """Forward pass of the policy network.

        Args:
            carry: LSTM carry state (h, c, n)
            x: Tuple of (observations, dones)
            rngs: Optional random number generators

        Returns:
            Tuple of (new_carry, policy_distribution)
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

        # Policy Head (Actor)
        actor_mean_val = self.actor_mean(lstm_output)
        pi = distrax.MultivariateNormalDiag(actor_mean_val, jnp.exp(self.actor_logstd.value))

        return new_carry, pi

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        """Initialize the LSTM carry state.

        Args:
            batch_size: Batch size
            hidden_dim: Hidden dimension size

        Returns:
            Tuple of (h_init, c_init, n_init) for hidden state, cell state, and normalizer state
        """
        return sLSTMBlock.initialize_carry(batch_size, hidden_dim)


# --- End Copied Components ---


# Define a TrainState for GRPO using NNX
class GRPOTrainState:
    """Training state for GRPO using NNX."""

    def __init__(
        self,
        *,
        model_state: Optional[nnx.State] = None,
        optimizer_state: Optional[optax.OptState] = None,
        graph_def: Optional[nnx.GraphDef] = None,
        rngs: Optional[nnx.Rngs] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.graph_def = graph_def
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.optimizer = optimizer

    @classmethod
    def create(cls, *, apply_fn=None, params=None, tx=None, **kwargs):  # apply_fn kept for API compatibility
        """Create a new instance with default values."""
        # This method is for compatibility with old code that expects TrainState
        # In NNX, we don't use apply_fn and params directly
        return cls(
            model_state=params,  # This would be the NNX state
            optimizer_state=tx.init(params) if tx is not None and params is not None else None,
            optimizer=tx,
            **kwargs,
        )

    def apply_gradients(self, *, grads=None):
        """Update the model state with gradients."""
        if grads is None or self.optimizer is None or self.optimizer_state is None or self.model_state is None:
            return self
        updates, new_optimizer_state = self.optimizer.update(grads, self.optimizer_state, self.model_state)
        new_model_state = nnx.state.apply_updates(self.model_state, updates)
        return self._replace(model_state=new_model_state, optimizer_state=new_optimizer_state)

    def _replace(self, **kwargs):
        """Return a new instance with the specified fields updated."""
        return type(self)(**{**self.__dict__, **kwargs})


# --- GRPO Specific Logic ---


class XLSTMGRPOTrainer:
    state: Optional[GRPOTrainState] = None
    # ref_state: Optional[GRPOTrainState] = None # Optional reference model state

    def __init__(
        self,
        env,  # Environment for context, might not be directly used in update step
        network: PolicyNetworkXLSTM,
        learning_rate: float = 5e-5,  # Often lower for preference methods
        beta: float = 0.1,  # DPO/GRPO temperature parameter
        batch_size: int = 4,  # Number of prompts per batch
        group_size: int = 4,  # Number of responses per prompt (k)
        n_epochs: int = 1,  # Typically fewer epochs for preference data
        seed: int = 42,
        # Add other relevant hyperparameters
    ):
        self.env = env  # Store for potential use (e.g., state/action space info)
        self.network = network
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.group_size = group_size  # k in GRPO
        self.n_epochs = n_epochs
        self.key = jax.random.PRNGKey(seed)

        # Reference model (optional, like DPO). For simplicity, start without explicit ref model.
        # self.ref_network = PolicyNetworkXLSTM(...) # Could be same arch
        # self.ref_params = None # Load separately or copy initial params

        self._initialize_model()

    def _initialize_model(self):
        # Similar init to PPO, but only for the policy network
        obs_space = self.env.observation_space
        action_space = self.env.action_space  # Needed for action dim

        if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise ValueError("Currently only supports Box observation and action spaces")

        obs_shape = obs_space.shape
        action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1

        if self.network.action_dim != action_dim:
            raise ValueError(
                f"Network action_dim ({self.network.action_dim}) does not match env action dim ({action_dim})"
            )

        dummy_obs_shape = (1,) + obs_shape
        dummy_obs = jnp.zeros(dummy_obs_shape, dtype=obs_space.dtype)
        initial_carry = self.network.initialize_carry(1, self.network.hidden_dim)
        dummy_input = (dummy_obs, jnp.zeros((1,)))  # (obs, done)

        self.key, init_key = jax.random.split(self.key)

        @jax.jit
        def init_network(key, carry, inp, network_instance):
            return network_instance.init(key, carry, inp)

        # Initialize the network
        _ = init_network(init_key, initial_carry, dummy_input, self.network)

        # If using a reference model, initialize/load its params here
        # ref_params = ...

        self.optimizer = optax.adam(learning_rate=self.learning_rate)

        # Create training state with NNX components
        # Initialize dummy state for now
        # Use a fixed seed for reproducibility
        self.state = GRPOTrainState(optimizer=self.optimizer, rngs=nnx.Rngs(params=42))

    # Function to generate a group of k action sequences for a given starting state/obs
    # This requires running the policy network rollout k times, potentially with sampling
    def _generate_group_sequences(self, *args, **kwargs):  # pylint: disable=unused-argument
        # TODO: Implement sequence generation logic using the policy
        # Needs to run network autoregressively for num_steps, k times
        # Returns: List or array of k action sequences and their log probabilities
        logger.warning("_generate_group_sequences is not implemented yet")
        return []  # Return empty list as placeholder

    # Function to evaluate and rank the k sequences based on some preference criteria
    # (e.g., simulated PnL, Sharpe ratio, risk metric)
    def _rank_sequences(self, sequences: List[Dict]) -> List[Dict]:
        # TODO: Implement preference scoring and sorting logic
        # Input: List of generated sequences (actions, states, etc.)
        # Output: Sorted list of sequences based on preference score
        logger.warning("_rank_sequences is a placeholder and returns input sequences unsorted.")
        return sequences  # Add return statement

    # GRPO Loss Function (needs careful implementation based on the paper)
    @staticmethod
    @partial(jax.jit, static_argnames=("apply_func", "beta", "network_hidden_dim"))
    def _grpo_loss_jit(params: nnx.State, batch: Dict, beta: float, apply_func: Callable, network_hidden_dim: int):
        # batch likely contains:
        #   - initial_obs (batch_size, obs_dim) # Observation before the sequence starts
        #   - initial_hidden_state (batch_size, carry_shape...) # Hidden state before the sequence starts
        #   - ranked_action_sequences (batch_size, group_size, seq_len, action_dim)
        #   - observations_seq (batch_size, group_size, seq_len, obs_dim) # Observations corresponding to each action step
        #   - dones_seq (batch_size, group_size, seq_len) # Done flags corresponding to each action step
        #   - ref_log_probs (optional, batch_size, group_size) # Precomputed log probs from reference model

        # initial_obs is needed if the scan starts from a single point, but here we need obs for each step
        # initial_hidden_state is crucial
        initial_hidden_state = batch["initial_hidden_state"]
        ranked_action_sequences = batch["ranked_action_sequences"]  # Shape (B, K, T, A)
        observations_seq = batch["observations_seq"]  # Shape (B, K, T, O)
        dones_seq = batch["dones_seq"]  # Shape (B, K, T)
        # Assuming ref_log_probs are not provided or are handled externally if needed

        batch_size, group_size, seq_len, _ = ranked_action_sequences.shape
        chex.assert_shape(observations_seq, (batch_size, group_size, seq_len, -1))
        chex.assert_shape(dones_seq, (batch_size, group_size, seq_len))
        # chex.assert_shape(initial_hidden_state, (batch_size, ...)) # Exact shape depends on LSTM carry

        # --- Function to compute log prob of a single sequence ---
        def compute_sequence_log_prob(p, init_carry, obs_steps, action_steps, done_steps):
            # p: model params
            # init_carry: hidden state before first step (h, c, n)
            # obs_steps: sequence of observations (T, O)
            # action_steps: sequence of actions taken (T, A)
            # done_steps: sequence of done flags (T,)

            chex.assert_rank([obs_steps, action_steps, done_steps], [2, 2, 1])
            chex.assert_axis_dimension(obs_steps, 0, seq_len)
            chex.assert_axis_dimension(action_steps, 0, seq_len)
            chex.assert_axis_dimension(done_steps, 0, seq_len)

            def scan_body(carry, xs):
                # carry: hidden state from previous step
                # xs: tuple (obs_step, action_step, done_step)
                obs_step, action_step, done_step = xs
                # Get shape info for debugging if needed
                chex.assert_rank([obs_step, action_step, done_step], [1, 1, 0])  # Shape for a single step in sequence

                # --- Apply Reset Logic ---
                reset_carry = PolicyNetworkXLSTM.initialize_carry(1, network_hidden_dim)  # Batch size 1 here
                h_prev, c_prev, n_prev = carry
                h_reset, c_reset, n_reset = reset_carry
                # done_step is scalar here, broadcast needed if carry components are not scalar
                done_broadcast = done_step  # Assuming carry components are (hidden_dim,) requires broadcasting later
                # Simple broadcast assumption for now, might need adjustment based on carry shape
                h_eff = jnp.where(done_broadcast, h_reset[0], h_prev)
                c_eff = jnp.where(done_broadcast, c_reset[0], c_prev)
                n_eff = jnp.where(done_broadcast, n_reset[0], n_prev)
                effective_carry = (h_eff, c_eff, n_eff)
                # --- End Reset Logic ---

                # Apply network for one step, get policy pi for the *next* action based on obs_step
                next_hidden_state, pi = apply_func(
                    {"params": p}, effective_carry, (obs_step[None, :], done_step[None,])
                )  # Add batch dim for apply_fn

                # Calculate log_prob of the *actual* action taken at this step
                log_prob_step = pi.log_prob(action_step[None, :])  # Add batch dim for log_prob

                # Return new state and the log_prob for this step
                return next_hidden_state, log_prob_step[0]  # Squeeze batch dim from log_prob

            # Prepare scan inputs: (obs_steps, action_steps, done_steps) for seq_len steps
            scan_inputs = (obs_steps, action_steps, done_steps)
            # Run scan
            _, log_probs_steps = jax.lax.scan(scan_body, init_carry, scan_inputs)
            # Sum log probs over the sequence length
            return jnp.sum(log_probs_steps)

        # --- End compute_sequence_log_prob ---

        # Vectorize the log prob calculation over the group_size (K) and batch_size (B) dimensions
        # Input shapes to vmapped function:
        # init_carry: (B, carry...) -> (B, K, carry...) [Needs broadcasting or tiling]
        # obs_steps: (B, K, T, O)
        # action_steps: (B, K, T, A)
        # done_steps: (B, K, T)

        # Tile initial hidden state to match (B, K, carry...)
        # This assumes initial_hidden_state has shape (B, carry...)
        tiled_initial_carry = jax.tree.map(
            lambda x: einops.repeat(x, "b ... -> b k ...", k=group_size), initial_hidden_state
        )

        # Vmap over Batch (B) and Group (K) dimensions simultaneously
        # Signature: (params, init_carry[b,k], obs[b,k,:,:], actions[b,k,:,:], dones[b,k,:]) -> log_prob[b,k]
        policy_log_probs = jax.vmap(
            jax.vmap(compute_sequence_log_prob, in_axes=(None, 0, 0, 0, 0)), in_axes=(None, 0, 0, 0, 0)
        )(params, tiled_initial_carry, observations_seq, ranked_action_sequences, dones_seq)
        chex.assert_shape(policy_log_probs, (batch_size, group_size))

        # Reference log probabilities (assuming 0 if no reference model)
        # If ref_log_probs were precomputed and passed in batch, use those.
        # Otherwise, set to zero.
        ref_log_probs = batch.get("ref_log_probs", jnp.zeros_like(policy_log_probs))
        chex.assert_shape(ref_log_probs, (batch_size, group_size))

        # Compute GRPO loss using pairwise comparisons within the ranked group
        total_loss = jnp.array(0.0)  # Use jnp.array for type consistency
        num_pairs = 0

        # Iterate through all pairs (i, j) where i is preferred over j (i < j in ranked list)
        for i in range(group_size):
            for j in range(i + 1, group_size):
                # Select log probs for pair (i, j) across the batch
                log_prob_policy_i = policy_log_probs[:, i]
                log_prob_policy_j = policy_log_probs[:, j]
                log_prob_ref_i = ref_log_probs[:, i]
                log_prob_ref_j = ref_log_probs[:, j]

                # Calculate implicit rewards r = beta * (log_pi - log_ref)
                diff_i = log_prob_policy_i - log_prob_ref_i
                diff_j = log_prob_policy_j - log_prob_ref_j

                # Loss for the pair (i preferred over j)
                logit = beta * (diff_i - diff_j)
                pair_loss = -jax.nn.log_sigmoid(logit)
                # Average loss over the batch dimension for this pair
                pair_loss_mean = jnp.mean(pair_loss)
                total_loss = total_loss + pair_loss_mean  # Use addition operator for JAX arrays
                num_pairs += 1

        # Average loss over all pairs
        if num_pairs > 0:
            # Convert to float to avoid type issues
            final_loss = float(total_loss / num_pairs)
        else:
            final_loss = 0.0  # Should not happen if group_size > 1

        # Info dict can include average rewards, log probs etc. if needed
        info = {"grpo_loss": final_loss, "avg_policy_log_prob": jnp.mean(policy_log_probs)}
        return final_loss, info

    @staticmethod
    @partial(jax.jit, static_argnames=("apply_func", "beta", "network_hidden_dim"))
    def _update_step_jit(
        state: GRPOTrainState, batch: Dict, beta: float, apply_func: Callable, network_hidden_dim: int
    ):
        if state.model_state is None:
            return state, 0.0, {}

        # Extract params from model_state
        params = state.model_state.filter(nnx.Param)

        grad_fn = jax.value_and_grad(
            lambda p, b, bt, af, nhd: XLSTMGRPOTrainer._grpo_loss_jit(p, b, bt, af, nhd), has_aux=True
        )
        (loss, info), grads = grad_fn(params, batch, beta, apply_func, network_hidden_dim)
        state = state.apply_gradients(grads=grads)
        return state, loss, info

    def train(self, preference_data: List[Dict], total_updates: int):
        # Preference data should be structured like:
        # [ { "prompt_obs": ..., "prompt_hidden_state": ..., "ranked_sequences": [seq1_actions, seq2_actions, ...] }, ... ]
        # where sequences are ranked from best to worst.

        if self.state is None:
            raise RuntimeError("Trainer state not initialized.")

        num_samples = len(preference_data)
        # For NNX, we need a way to call the model
        if self.state is None:
            raise RuntimeError("Trainer state not initialized")

        # Create a simple function to call the model
        # This avoids direct access to graph_def.merge which might not be available
        # _params is unused but kept for API compatibility
        apply_func = lambda _params, carry, inputs: self.network(carry, inputs)


# Wrapper class that inherits from BaseModel to integrate with the ensemble
class XLSTMGRPOAgent(BaseModel):
    """xLSTM GRPO Agent that inherits from BaseModel for ensemble integration."""

    def __init__(self, name: str = "xlstm_grpo", target_column: str = "close", horizon: int = 5, **kwargs):
        """Initialize the xLSTM GRPO Agent.

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
        self.learning_rate = kwargs.get("learning_rate", 5e-5)  # Lower learning rate for GRPO
        self.batch_size = kwargs.get("batch_size", 4)
        self.group_size = kwargs.get("group_size", 4)  # Number of responses per prompt (k)
        self.n_epochs = kwargs.get("n_epochs", 1)
        self.beta = kwargs.get("beta", 0.1)  # GRPO temperature parameter
        self.seed = kwargs.get("seed", 42)

        # Initialize model state
        self.model_state = None
        self.graph_def = None
        self.trainer = None
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "XLSTMGRPOAgent":
        """Fit the xLSTM GRPO model to training data.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            # Extract parameters from kwargs
            total_updates = kwargs.get("total_updates", 1000)

            # Log training parameters
            logger.info(f"Training xLSTM GRPO model with {total_updates} updates")

            # TODO: Implement actual training logic using the XLSTMGRPOTrainer
            # For now, just set is_fitted to True for testing
            self.is_fitted = True

            return self
        except Exception as e:
            logger.error(f"Error fitting xLSTM GRPO model: {e}")
            return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted xLSTM GRPO model.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            logger.warning("xLSTM GRPO model not fitted yet")
            return np.array([])

        try:
            # TODO: Implement actual prediction logic
            # For now, return random predictions for testing
            np.random.seed(self.seed)
            predictions = np.random.uniform(-1.0, 1.0, size=len(X))

            return predictions
        except Exception as e:
            logger.error(f"Error predicting with xLSTM GRPO model: {e}")
            return np.array([])

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Save model to disk.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        save_dir = directory / "xlstm_grpo"
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
                "batch_size": self.batch_size,
                "group_size": self.group_size,
                "n_epochs": self.n_epochs,
                "beta": self.beta,
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
                "seed": self.seed,
            }

            with open(model_dir / "config.pkl", "wb") as f:
                pickle.dump(config, f)

            logger.info(f"xLSTM GRPO model saved to {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"Error saving xLSTM GRPO model: {e}")
            return model_dir

    def load(self, model_path: Path) -> "XLSTMGRPOAgent":
        """Load model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            config_path = model_path / "config.pkl"

            if not config_path.exists():
                logger.error(f"Cannot load xLSTM GRPO model: Missing config file in {model_path}")
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
            self.batch_size = config["batch_size"]
            self.group_size = config["group_size"]
            self.n_epochs = config["n_epochs"]
            self.beta = config["beta"]
            self.feature_names = config.get("feature_names", [])
            self.seed = config.get("seed", 42)
            self.is_fitted = config["is_fitted"]

            logger.info(f"xLSTM GRPO model loaded from {model_path}")
            return self
        except Exception as e:
            logger.error(f"Error loading xLSTM GRPO model from {model_path}: {e}")
            self.is_fitted = False
            return self
