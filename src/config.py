# src/config.py
"""Configuration settings for the time series ensemble model."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import dotenv

dotenv.load_dotenv()

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MLRUNS_DIR = PROJECT_DIR / "mlruns"
LOGS_DIR = PROJECT_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API keys
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# MLflow tracking URI — defaults to a local file store under the project root.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")


def resolve_jax_device(preference: str = "auto") -> str:
    """Resolve a JAX device preference to a concrete backend name.

    "auto" picks gpu/tpu when available, otherwise cpu. Explicit values pass through.
    """
    if preference != "auto":
        return preference
    try:
        import jax

        backends = {d.platform.lower() for d in jax.devices()}
        if backends & {"gpu", "cuda", "rocm"}:
            return "gpu"
        if "tpu" in backends:
            return "tpu"
        return "cpu"
    except Exception:
        return "cpu"


# JAX configuration
JAX_CONFIG = {
    "jit": True,
    "device": "auto",  # resolved via resolve_jax_device() at use sites
    "precision": "float32",
    "memory_fraction": 0.8,
}


@dataclass
class ModelConfig:
    """Configuration for an individual model in the ensemble."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    params: Optional[dict] = None
    jax_config: Optional[dict] = None


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""

    models: List[ModelConfig]
    weighting_strategy: str = "static"  # "static", "dynamic", "adaptive"
    refit_interval: int = 0
    optimize_weights: bool = False
    # Vol-targeting knobs for the forecast→position mapping. Exposed here
    # (Phase 2.7) so a hyperparameter sweep can vary them as first-class,
    # persisted config rather than poking post-construction attrs.
    target_vol: float = 1.0
    position_cap: float = 1.0

    def __post_init__(self):
        assert self.target_vol > 0, f"target_vol must be > 0; got {self.target_vol}"
        assert self.position_cap > 0, (
            f"position_cap must be > 0; got {self.position_cap}"
        )


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Phase 2.2 retired ``train_test_split`` and ``cv_folds`` — train/test
    folding is now driven entirely by the WFO knobs (``n_splits``,
    ``purge_horizon``, ``embargo_pct``, ``expanding``) consumed by
    ``PurgedWalkForward`` in scripts/training.py and scripts/backtest.py.
    """

    symbols: List[str]
    timeframe: str = "1d"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    prediction_horizon: int = 5
    use_sentiment: bool = False
    optimize: bool = False
    # Outer-WFO knobs. purge_horizon=None -> use prediction_horizon.
    n_splits: int = 5
    purge_horizon: Optional[int] = None
    embargo_pct: float = 0.01
    expanding: bool = True

    def __post_init__(self):
        assert self.prediction_horizon > 0, (
            f"prediction_horizon must be > 0; got {self.prediction_horizon}"
        )
        assert self.n_splits >= 2, f"n_splits must be >= 2; got {self.n_splits}"
        assert 0.0 <= self.embargo_pct < 1.0, (
            f"embargo_pct must be in [0, 1); got {self.embargo_pct}"
        )
        if self.purge_horizon is not None:
            assert self.purge_horizon >= 0, (
                f"purge_horizon must be >= 0; got {self.purge_horizon}"
            )

    @property
    def effective_purge_horizon(self) -> int:
        """Purge horizon for the WFO splitter, defaulting to prediction_horizon
        when the explicit override is None — the forward-return label window
        is exactly ``prediction_horizon`` bars wide so that's the right
        default for AFML §7.4-style purging."""
        return self.purge_horizon if self.purge_horizon is not None else self.prediction_horizon


@dataclass
class ExecutionConfig:
    """Execution model parameters: fills, costs, borrow.

    bps = basis points (1 bp = 0.01%). All cost figures one-way unless noted.
    """

    # Half of the quoted bid-ask spread. One-way cost on every fill.
    spread_bps: float = 1.0
    # Linear price-impact coefficient: extra slip_bps = slippage_coeff * |notional| / portfolio_value.
    # Coeff of 10 means a 100%-of-portfolio trade incurs +10 bps on top of half-spread.
    slippage_coeff: float = 10.0
    # Commission charged as bps of traded notional (one-way).
    commission_bps: float = 1.0
    # Annualized borrow rate charged on |short notional|, accrued daily over a 252-day year.
    borrow_rate_bps_annual: float = 50.0
    # "MOO" (market-on-open, fills at next bar's open) or "MOC" (next bar's close).
    # MOO is the rigorous default; close-of-bar t signal -> fill at open of t+1.
    default_order_type: str = "MOO"
    # Trading days per year used for borrow accrual.
    trading_days_per_year: int = 252

    def __post_init__(self):
        assert self.spread_bps >= 0, f"spread_bps must be >= 0; got {self.spread_bps}"
        assert self.slippage_coeff >= 0, (
            f"slippage_coeff must be >= 0; got {self.slippage_coeff}"
        )
        assert self.commission_bps >= 0, (
            f"commission_bps must be >= 0; got {self.commission_bps}"
        )
        assert self.borrow_rate_bps_annual >= 0, (
            f"borrow_rate_bps_annual must be >= 0; got {self.borrow_rate_bps_annual}"
        )
        assert self.default_order_type in ("MOO", "MOC"), (
            f"default_order_type must be MOO or MOC; got {self.default_order_type}"
        )
        assert self.trading_days_per_year > 0, (
            f"trading_days_per_year must be > 0; got {self.trading_days_per_year}"
        )


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""

    initial_capital: float = 10000.0
    position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.05
    risk_free_rate: float = 0.02
    # Minimum |position_target| to emit a LONG/SHORT signal (else FLAT).
    # Exposed as config (Phase 2.7) so a sweep can vary the trade threshold.
    signal_threshold: float = 0.1
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self):
        assert 0 < self.position_size <= 1, (
            f"position_size must be in (0, 1]; got {self.position_size}"
        )
        assert 0 <= self.stop_loss < 1, f"stop_loss must be in [0, 1); got {self.stop_loss}"
        assert self.initial_capital > 0, (
            f"initial_capital must be > 0; got {self.initial_capital}"
        )
        assert 0 <= self.signal_threshold < 1, (
            f"signal_threshold must be in [0, 1); got {self.signal_threshold}"
        )


# Single source of truth for ensemble member weights. Scripts must read from
# here rather than hardcoding per-model overrides.
DEFAULT_MODEL_WEIGHTS = {
    "arima": 1.0,
    "prophet": 1.0,
    "lstm": 1.0,
    "xgboost": 1.0,
    "lstm_ppo": 1.0,
    "xlstm_ppo": 1.0,
    "xlstm_grpo": 1.0,
}

DEFAULT_MODELS = [
    ModelConfig(name=name, enabled=True, weight=weight)
    for name, weight in DEFAULT_MODEL_WEIGHTS.items()
]

DEFAULT_ENSEMBLE_CONFIG = EnsembleConfig(
    models=DEFAULT_MODELS,
    weighting_strategy="dynamic",
    refit_interval=0,
    optimize_weights=False,
)

DEFAULT_TRADING_CONFIG = TradingConfig(
    initial_capital=10000.0,
    position_size=0.1,
    stop_loss=0.02,
    take_profit=0.05,
    risk_free_rate=0.02,
)

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    symbols=["AAPL", "MSFT", "GOOG", "AMZN"],
    timeframe="1d",
    start_date="2020-01-01",
    end_date=None,
    prediction_horizon=5,
    use_sentiment=False,
    optimize=False,
)
