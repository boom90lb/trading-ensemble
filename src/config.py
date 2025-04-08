# src/config.py
"""Configuration settings for the time series ensemble model."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API keys
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


@dataclass
class ModelConfig:
    """Configuration for individual model in the ensemble."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    params: Optional[dict] = None


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""

    models: List[ModelConfig]
    weighting_strategy: str = "static"  # Options: "static", "dynamic", "adaptive"
    refit_interval: int = 0  # Number of periods between refits (0 = no refit)
    optimize_weights: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    symbols: List[str]
    timeframe: str = "1d"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    train_test_split: float = 0.8
    prediction_horizon: int = 5
    use_sentiment: bool = False
    optimize: bool = False
    cv_folds: int = 5


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""

    initial_capital: float = 10000.0
    position_size: float = 0.1  # Percentage of portfolio per position
    stop_loss: float = 0.02  # Stop loss as percentage of entry price
    take_profit: float = 0.05  # Take profit as percentage of entry price
    commission: float = 0.001  # Commission as percentage
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe ratio


# Default configurations
DEFAULT_MODEL_WEIGHTS = {
    "arima": 1.0,
    "prophet": 1.0,
    "lstm": 1.0,
    "xgboost": 1.0,
    "lstm_ppo": 2.0,  # Higher weight for RL model
}

DEFAULT_MODELS = [
    ModelConfig(name="arima", enabled=True, weight=DEFAULT_MODEL_WEIGHTS["arima"]),
    ModelConfig(name="prophet", enabled=True, weight=DEFAULT_MODEL_WEIGHTS["prophet"]),
    ModelConfig(name="lstm", enabled=True, weight=DEFAULT_MODEL_WEIGHTS["lstm"]),
    ModelConfig(name="xgboost", enabled=True, weight=DEFAULT_MODEL_WEIGHTS["xgboost"]),
    ModelConfig(name="lstm_ppo", enabled=True, weight=DEFAULT_MODEL_WEIGHTS["lstm_ppo"]),
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
    commission=0.001,
    risk_free_rate=0.02,
)

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    symbols=["AAPL", "MSFT", "GOOG", "AMZN"],
    timeframe="1d",
    start_date="2020-01-01",
    end_date=None,
    train_test_split=0.8,
    prediction_horizon=5,
    use_sentiment=False,
    optimize=False,
    cv_folds=5,
)
