#!/usr/bin/env python
"""Script for training the time series ensemble model."""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd  # Import pandas to access the error type

# Suppress specific Pandas warnings (like DataFrame fragmentation)
try:
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
except AttributeError:
    # Older pandas versions might not have this specific error type exposed
    # You might need a more general filter or upgrade pandas if this is critical
    warnings.warn("Could not specifically filter PerformanceWarning. Filtering FutureWarning instead.", FutureWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

# Configure TF log level *before* importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ERROR level
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# Configure base logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silence Prophet logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import pandas as pd

from src.config import DEFAULT_TRAINING_CONFIG, EnsembleConfig, ModelConfig, TrainingConfig
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.sentiment_analysis import SentimentAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the time series ensemble model.")

    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_TRAINING_CONFIG.symbols),
        help="Comma-separated list of symbols to train on",
    )

    parser.add_argument(
        "--timeframe", type=str, default=DEFAULT_TRAINING_CONFIG.timeframe, help="Time interval (e.g., 1d, 1h)"
    )

    parser.add_argument(
        "--start_date", type=str, default=DEFAULT_TRAINING_CONFIG.start_date, help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument("--end_date", type=str, default=None, help="End date in YYYY-MM-DD format")

    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")

    parser.add_argument("--use_sentiment", action="store_true", help="Use sentiment analysis")

    parser.add_argument(
        "--models",
        type=str,
        default="arima,prophet,lstm,xgboost,lstm_ppo",
        help="Comma-separated list of models to use in the ensemble",
    )

    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")

    parser.add_argument("--rl_timesteps", type=int, default=100000, help="Number of timesteps for RL training")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse symbols
    symbols = args.symbols.split(",")
    symbols_str = "_".join(symbols)  # Create a string representation for the directory name

    # --- Create unique output directory for this run ---
    run_name = f"run_{symbols_str}_{timestamp}"
    output_dir = Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving run artifacts to: {output_dir.resolve()}")
    # ---

    # Create training config
    training_config = TrainingConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=args.horizon,
    )

    logger.info(f"Training configuration: {training_config}")

    # Initialize data loader
    data_loader = DataLoader()

    # Fetch data
    logger.info(f"Fetching data for {symbols}")
    all_data, train_test_splits = data_loader.fetch_and_prepare_training_data(training_config)

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Initialize sentiment analyzer if requested
    sentiment_analyzer = None
    if args.use_sentiment:
        logger.info("Initializing sentiment analyzer")
        sentiment_analyzer = SentimentAnalyzer()

    # Create enhanced datasets with features and sentiment
    enhanced_data = {}
    for symbol, (train_df, test_df) in train_test_splits.items():
        logger.info(f"Creating features for {symbol}")

        # Create technical features
        train_with_features = feature_engineer.create_features(train_df)
        test_with_features = feature_engineer.create_features(test_df)

        # Create lagged features
        train_with_features = feature_engineer.create_lagged_features(train_with_features, [1, 2, 5, 10])
        test_with_features = feature_engineer.create_lagged_features(test_with_features, [1, 2, 5, 10])

        # Create target variable
        train_with_features = feature_engineer.create_target_variable(
            train_with_features, "close", training_config.prediction_horizon
        )
        test_with_features = feature_engineer.create_target_variable(
            test_with_features, "close", training_config.prediction_horizon
        )

        # Add sentiment features if requested
        if args.use_sentiment and sentiment_analyzer:
            logger.info(f"Creating sentiment features for {symbol}")
            train_sentiment = sentiment_analyzer.create_sentiment_features(
                symbol, pd.DatetimeIndex(train_with_features.index)  # type: ignore
            )
            test_sentiment = sentiment_analyzer.create_sentiment_features(
                symbol, pd.DatetimeIndex(test_with_features.index)  # type: ignore
            )

            # Join sentiment features
            if not train_sentiment.empty:
                train_with_features = train_with_features.join(train_sentiment)
            if not test_sentiment.empty:
                test_with_features = test_with_features.join(test_sentiment)

        # Fit scalers on training data
        feature_engineer.fit_scalers(train_with_features, symbol)

        # Scale features
        train_scaled = feature_engineer.transform_features(train_with_features, symbol, is_train=True)
        test_scaled = feature_engineer.transform_features(test_with_features, symbol, is_train=False)

        # Store enhanced data
        enhanced_data[symbol] = {"train": train_scaled, "test": test_scaled}

    # Parse model list
    model_names = args.models.split(",")

    # Create ensemble config
    model_configs = []
    for name in model_names:
        if name.strip():
            # Higher weight for RL model
            weight = 2.0 if name == "lstm_ppo" else 1.0
            model_configs.append(ModelConfig(name=name, enabled=True, weight=weight))

    ensemble_config = EnsembleConfig(models=model_configs, weighting_strategy="dynamic")

    # Create and train ensemble model
    logger.info("Creating ensemble model")
    ensemble = EnsembleModel(target_column="close", horizon=training_config.prediction_horizon, config=ensemble_config)

    # Train for each symbol
    all_run_metrics = {}  # Store metrics for all symbols in this run
    for symbol, data_dict in enhanced_data.items():
        logger.info(f"Training ensemble for {symbol}")

        train_df = data_dict["train"]
        test_df = data_dict["test"]

        # Prepare X and y
        target_col = f"target_{training_config.prediction_horizon}"

        # Drop rows with NaN in target
        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        # Split features and target
        X_train = train_df.drop(columns=[col for col in train_df.columns if "target_" in col or "direction_" in col])
        y_train = train_df[target_col]

        X_test = test_df.drop(columns=[col for col in test_df.columns if "target_" in col or "direction_" in col])
        y_test = test_df[target_col]

        # Train the ensemble models
        logger.info(f"Training ensemble models for {symbol}")
        try:
            # Pass rl_timesteps for LSTMPPO if it's in the models
            fit_kwargs = {}
            if "lstm_ppo" in model_names:
                fit_kwargs["lstm_ppo_timesteps"] = args.rl_timesteps
                # Pass required features for LSTMPPO
                fit_kwargs["lstm_ppo_features"] = [
                    col for col in X_train.columns if not (col.startswith("target_") or col.startswith("direction_"))
                ]
                # Pass original close price needed for LSTMPPO fitting/environment
                fit_kwargs["lstm_ppo_X_train_with_close"] = train_df  # Pass the unscaled df

            ensemble.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            logger.error(f"Error training ensemble for {symbol}: {e}")
            continue

        # Evaluate on test data
        logger.info(f"Evaluating ensemble for {symbol}")
        metrics = ensemble.evaluate(X_test, y_test)
        all_run_metrics[symbol] = metrics  # Store metrics for this symbol

        logger.info(f"Test metrics for {symbol}: {metrics}")

        # Get model contributions
        contributions = ensemble.get_model_contributions(X_test.iloc[:5])
        logger.info(f"Model contributions for {symbol}:\n{contributions}")

        # --- Save individual symbol results to the run directory ---
        symbol_results_path = output_dir / f"results_{symbol}.json"
        # No need to create parent dir, output_dir already exists
        with open(symbol_results_path, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "metrics": metrics,
                    "weights": {k: float(v) for k, v in ensemble.weights.items()},
                    "timestamp": timestamp,  # Keep original run timestamp
                },
                f,
                indent=2,
            )
        logger.info(f"Saved evaluation results for {symbol} to {symbol_results_path}")
        # ---

    # --- Save the final ensemble model to the run directory ---
    logger.info("Saving ensemble model state")
    ensemble_save_path = output_dir / "ensemble_model"  # Pass a sub-directory
    ensemble.save(directory=ensemble_save_path)  # Pass the dedicated directory
    logger.info(f"Ensemble model state saved to {ensemble_save_path}")
    # ---

    # --- Optionally, save aggregated metrics for the whole run ---
    aggregated_results_path = output_dir / "aggregated_results.json"
    with open(aggregated_results_path, "w") as f:
        json.dump(all_run_metrics, f, indent=2)
    logger.info(f"Saved aggregated evaluation results to {aggregated_results_path}")
    # ---

    logger.info("Training complete")


if __name__ == "__main__":
    main()
