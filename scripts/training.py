#!/usr/bin/env python
"""Script for training the time series ensemble model."""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # Import pandas to access the error type

from src.config import DEFAULT_TRAINING_CONFIG, EnsembleConfig, ModelConfig, TrainingConfig
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.sentiment_analysis import SentimentAnalyzer

# Suppress specific Pandas warnings (like DataFrame fragmentation)
try:
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
except AttributeError:
    # Older pandas versions might not have this specific error type exposed
    # You might need a more general filter or upgrade pandas if this is critical
    warnings.warn("Could not specifically filter PerformanceWarning. Filtering FutureWarning instead.", FutureWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

# Configure TF log level *before* importing tensorflow
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ERROR level
# import tensorflow as tf

# tf.get_logger().setLevel("ERROR")

# Configure base logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silence Prophet logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


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

    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")

    parser.add_argument(
        "--checkpoint_freq", type=int, default=50000, help="Frequency for saving checkpoints during RL training"
    )

    parser.add_argument("--extended_features", action="store_true", help="Use extended feature set for training")

    return parser.parse_args()


def select_rl_features(X_train: pd.DataFrame) -> list:
    """Select optimal features for the LSTM-PPO model.

    Args:
        X_train: Training features DataFrame

    Returns:
        List of selected feature names
    """
    # Priority features that are important for trading decisions
    priority_features = [
        # Base price data
        "open",
        "high",
        "low",
        "close",
        "volume",
        # Moving averages
        "ma5",
        "ma20",
        "ema5",
        "ema20",
        # Momentum and trend
        "macd",
        "macd_hist",
        "rsi_14",
        "price_change",
        "price_change_5",
        "price_change_10",
        # Volatility
        "volatility_5",
        "volatility_20",
        "bb_width",
        "atr_14",
        # Volume indicators
        "volume_change",
        "volume_ma5",
        "mfi_14",
        # Other technical indicators
        "daily_range",
        "gap",
        "roc_5",
        "roc_10",
    ]

    # Check which of the priority features are actually in the dataframe
    available_features = [f for f in priority_features if f in X_train.columns]

    # If we have very few features, add some lagged versions
    if len(available_features) < 10:
        # Find all lag features for the base features we do have
        for col in [c for c in X_train.columns if any(c.startswith(f) for f in ["close", "rsi", "macd", "price"])]:
            lag_features = [f for f in X_train.columns if f.startswith(f"{col}_lag")]
            # Add the most recent lags (lag1, lag2)
            recent_lags = [f for f in lag_features if f.endswith("lag1") or f.endswith("lag2")]
            available_features.extend(recent_lags)

    # Limit to at most 30 features to prevent overfitting
    if len(available_features) > 30:
        # Keep the most important indicators
        essential = [
            f
            for f in available_features
            if f in ["close", "price_change", "rsi_14", "macd", "bb_width", "volatility_20"]
        ]

        # Keep some volume indicators
        volume = [f for f in available_features if "volume" in f][:3]

        # Keep some moving averages
        mas = [f for f in available_features if f.startswith("ma") or f.startswith("ema")][:4]

        # Additional features based on importance
        additional = [f for f in available_features if f not in essential + volume + mas]
        additional = sorted(additional, key=lambda x: priority_features.index(x) if x in priority_features else 999)[
            : 30 - len(essential) - len(volume) - len(mas)
        ]

        available_features = essential + volume + mas + additional

    logger.info(f"Selected {len(available_features)} features for LSTM-PPO model")
    return available_features


def check_gpu_availability():
    """Check if GPU is available for TensorFlow and PyTorch."""
    # Check TensorFlow GPU
    # tf_gpus = tf.config.list_physical_devices("GPU")
    # tf_gpu_available = len(tf_gpus) > 0
    tf_gpu_available = False  # Assume false now that we removed TF

    # Check PyTorch GPU
    try:
        import torch

        torch_gpu_available = torch.cuda.is_available()
        torch_gpu_name = torch.cuda.get_device_name(0) if torch_gpu_available else "None"
    except ImportError:
        torch_gpu_available = False
        torch_gpu_name = "Torch not installed"

    # Log results
    if tf_gpu_available:
        # This block will now likely never run, but keep structure for potential future re-integration if needed.
        # logger.info(f"TensorFlow GPU available: {len(tf_gpus)} device(s)")
        # for i, gpu in enumerate(tf_gpus):
        #     logger.info(f"  GPU {i}: {gpu.name}")
        pass  # Keep the block but do nothing as tf_gpu_available is hardcoded false
    else:
        logger.warning("TensorFlow GPU check removed. Relying on PyTorch check.")

    if torch_gpu_available:
        logger.info(f"PyTorch GPU available: {torch_gpu_name}")
    else:
        logger.warning("PyTorch GPU not available, using CPU")

    return tf_gpu_available or torch_gpu_available


def clean_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Perform additional data cleaning for training.

    Args:
        df: DataFrame with features

    Returns:
        Cleaned DataFrame
    """
    result = df.copy()

    # Replace infinities with NaN
    result = result.replace([np.inf, -np.inf], np.nan)

    # Check for columns with too many NaNs (over 30%)
    nan_percentage = result.isna().mean()
    high_nan_cols = nan_percentage[nan_percentage > 0.3].index.tolist()

    if high_nan_cols:
        logger.warning(f"Dropping {len(high_nan_cols)} columns with >30% NaN values")
        result = result.drop(columns=high_nan_cols)

    # Forward fill NaNs (using previous values)
    result = result.ffill()

    # Backward fill any remaining NaNs at the beginning
    result = result.bfill()

    # Fill any still remaining NaNs with zeros
    result = result.fillna(0)

    # Check for extreme outliers using IQR method
    for col in result.select_dtypes(include=["number"]).columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 5 * iqr  # 5x IQR for very extreme outliers only
        upper_bound = q3 + 5 * iqr

        # Count extreme outliers
        outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
        if outliers > 0:
            # Clip extreme values
            result[col] = result[col].clip(lower_bound, upper_bound)
            logger.debug(f"Clipped {outliers} extreme outliers in column {col}")

    return result


def main():
    """Main function."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check GPU availability if requested
    if args.gpu:
        gpu_available = check_gpu_availability()
        if not gpu_available:
            logger.warning("GPU acceleration requested but no GPU available")

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
        use_sentiment=args.use_sentiment,
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

        # Additional data cleaning for training
        train_with_features = clean_data_for_training(train_with_features)
        test_with_features = clean_data_for_training(test_with_features)

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

        # --- Ensure X_test has the same columns as X_train ---
        # This is crucial if X_train columns were modified (e.g., by feature selection)
        # *after* the initial split.
        X_test = X_test.reindex(
            columns=X_train.columns, fill_value=0
        )  # Add missing cols, fill with 0 (or appropriate value)
        # ---

        # Train the ensemble models
        logger.info(f"Training ensemble models for {symbol}")
        try:
            # Prepare fit kwargs
            fit_kwargs = {}

            # Pass rl_timesteps for LSTMPPO if it's in the models
            if "lstm_ppo" in model_names:
                # Set RL training parameters
                fit_kwargs["lstm_ppo_timesteps"] = args.rl_timesteps
                fit_kwargs["lstm_ppo_checkpoint_freq"] = args.checkpoint_freq

                # Set device for RL model
                if args.gpu:
                    fit_kwargs["lstm_ppo_device"] = "cuda" if gpu_available else "cpu"
                else:
                    fit_kwargs["lstm_ppo_device"] = "cpu"

                # Improved feature selection for LSTM-PPO
                lstm_ppo_features = select_rl_features(X_train)
                fit_kwargs["lstm_ppo_features"] = lstm_ppo_features

                # Pass original close price needed for LSTMPPO fitting/environment
                fit_kwargs["lstm_ppo_X_train_with_close"] = train_df  # Pass the unscaled df

                # --- Pass symbol-specific TensorBoard path ---
                # Use the main run output directory as the base
                ppo_log_path = output_dir / f"tensorboard_logs_{symbol}"
                fit_kwargs["tensorboard_log_path"] = str(ppo_log_path)  # Pass as string
                # ---

                # Log selected features
                logger.info(
                    f"Selected {len(lstm_ppo_features)} features for LSTM-PPO model: "
                    f"{', '.join(lstm_ppo_features[:5])}..."
                )

                # Save selected features for reference
                with open(output_dir / f"lstm_ppo_features_{symbol}.json", "w") as f:
                    json.dump(lstm_ppo_features, f, indent=2)

            ensemble.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            logger.error(f"Error training ensemble for {symbol}: {e}", exc_info=True)
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

    # --- Save training configuration ---
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        # Convert training config to dict for JSON serialization
        config_dict = {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon": args.horizon,
            "use_sentiment": args.use_sentiment,
            "models": model_names,
            "rl_timesteps": args.rl_timesteps,
            "gpu": args.gpu,
            "timestamp": timestamp,
        }
        json.dump(config_dict, f, indent=2)
    # ---

    # --- Optionally, save aggregated metrics for the whole run ---
    aggregated_results_path = output_dir / "aggregated_results.json"
    with open(aggregated_results_path, "w") as f:
        # Convert numpy values to Python natives for JSON serialization
        serializable_metrics = {}
        for symbol, metrics in all_run_metrics.items():
            serializable_metrics[symbol] = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Saved aggregated evaluation results to {aggregated_results_path}")
    # ---

    logger.info("Training complete")


if __name__ == "__main__":
    main()
