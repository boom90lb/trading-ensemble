#!/usr/bin/env python
"""Script for training the time series ensemble model."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_TRAINING_CONFIG, EnsembleConfig, ModelConfig, TrainingConfig
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.models.lstm_ppo import LSTMPPO
from src.sentiment_analysis import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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

    # Parse symbols
    symbols = args.symbols.split(",")

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

        # Special handling for LSTM-PPO model
        if "lstm_ppo" in model_names:
            logger.info(f"Training LSTM-PPO for {symbol}")
            lstm_ppo = LSTMPPO(
                target_column="close",
                horizon=training_config.prediction_horizon,
                features=[
                    col for col in X_train.columns if not (col.startswith("target_") or col.startswith("direction_"))
                ],
                window_size=20,  # Adjust window size as needed
            )

            # For LSTM-PPO, we need to include the close price in X_train
            lstm_ppo_train = X_train.copy()
            lstm_ppo_train["close"] = train_df["close"] if "close" in train_df else y_train

            # Train LSTM-PPO model
            lstm_ppo.fit(lstm_ppo_train, y_train, total_timesteps=args.rl_timesteps)

            # Save LSTM-PPO model separately
            lstm_ppo.save()

            # Add to ensemble manually
            ensemble.models["lstm_ppo"] = lstm_ppo
            ensemble.is_fitted = True

        # Train the remaining ensemble models
        logger.info(f"Training remaining ensemble models for {symbol}")
        try:
            ensemble.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training ensemble for {symbol}: {e}")
            continue

        # Evaluate on test data
        logger.info(f"Evaluating ensemble for {symbol}")
        metrics = ensemble.evaluate(X_test, y_test)

        logger.info(f"Test metrics for {symbol}: {metrics}")

        # Get model contributions
        contributions = ensemble.get_model_contributions(X_test.iloc[:5])
        logger.info(f"Model contributions for {symbol}:\n{contributions}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"results/training_{symbol}_{timestamp}.json")
        results_path.parent.mkdir(exist_ok=True, parents=True)

        with open(results_path, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "metrics": metrics,
                    "weights": {k: float(v) for k, v in ensemble.weights.items()},
                    "timestamp": timestamp,
                },
                f,
                indent=2,
            )

    # Save the final ensemble model
    logger.info("Saving ensemble model")
    ensemble.save()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
