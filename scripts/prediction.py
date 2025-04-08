#!/usr/bin/env python
"""Script for making predictions with the time series ensemble model."""

import argparse
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR
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
    parser = argparse.ArgumentParser(description="Make predictions with the time series ensemble model.")

    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated list of symbols to predict")

    parser.add_argument("--timeframe", type=str, default="1d", help="Time interval (e.g., 1d, 1h)")

    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")

    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model (default: latest model)")

    parser.add_argument("--use_sentiment", action="store_true", help="Use sentiment analysis")

    parser.add_argument("--days", type=int, default=30, help="Number of days of historical data to use")

    parser.add_argument("--plot", action="store_true", help="Plot predictions")

    return parser.parse_args()


def load_model(path=None, horizon=5):
    """Load the ensemble model.

    Args:
        path: Path to the saved model (default: latest model)
        horizon: Forecast horizon

    Returns:
        Loaded ensemble model
    """
    # Find latest model if path not provided
    if path is None:
        model_paths = list(MODELS_DIR.glob(f"ensemble_h{horizon}*.pkl"))
        if not model_paths:
            logger.error(f"No ensemble model found for horizon {horizon}")
            raise FileNotFoundError(f"No ensemble model found for horizon {horizon}")
        path = max(model_paths, key=lambda p: p.stat().st_mtime)
    else:
        path = Path(path)
        if not path.exists():
            logger.error(f"Model file not found: {path}")
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load model
    logger.info(f"Loading model from {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)

    # Check if LSTM-PPO directory exists and load it
    lstm_ppo_dir = MODELS_DIR / f"lstm_ppo_h{horizon}"
    lstm_ppo_path = lstm_ppo_dir / "model.zip"

    if lstm_ppo_path.exists():
        logger.info(f"Loading LSTM-PPO model from {lstm_ppo_path}")
        try:
            lstm_ppo = LSTMPPO(target_column="close", horizon=horizon)
            lstm_ppo.load(lstm_ppo_path)

            # Add to ensemble
            if "lstm_ppo" not in model.models:
                model.models["lstm_ppo"] = lstm_ppo
                model.weights["lstm_ppo"] = 2.0  # Higher weight for RL model

                # Normalize weights
                total_weight = sum(model.weights.values())
                for k in model.weights:
                    model.weights[k] /= total_weight
        except Exception as e:
            logger.error(f"Error loading LSTM-PPO model: {e}")

    return model


def make_predictions(symbols, timeframe, horizon, model_path, use_sentiment, days, plot):
    """Make predictions for the given symbols.

    Args:
        symbols: List of symbols to predict
        timeframe: Time interval
        horizon: Forecast horizon
        model_path: Path to the saved model
        use_sentiment: Whether to use sentiment analysis
        days: Number of days of historical data to use
        plot: Whether to plot predictions
    """
    # Load model
    model = load_model(model_path, horizon)

    # Initialize data loader
    data_loader = DataLoader()

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Initialize sentiment analyzer if requested
    sentiment_analyzer = None
    if use_sentiment:
        logger.info("Initializing sentiment analyzer")
        sentiment_analyzer = SentimentAnalyzer()

    # Set date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Fetch latest data
    all_predictions = {}

    for symbol in symbols:
        logger.info(f"Making predictions for {symbol}")

        # Fetch data
        df = data_loader.fetch_historical_data(
            symbol=symbol, interval=timeframe, start_date=start_date, end_date=end_date
        )

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            continue

        # Create features
        df_with_features = feature_engineer.create_features(df)
        df_with_features = feature_engineer.create_lagged_features(df_with_features, [1, 2, 5, 10])

        # Add sentiment features if requested
        if use_sentiment and sentiment_analyzer:
            logger.info(f"Creating sentiment features for {symbol}")
            sentiment_features = sentiment_analyzer.create_sentiment_features(symbol, df_with_features.index)

            if not sentiment_features.empty:
                df_with_features = df_with_features.join(sentiment_features)

        # Prepare input features
        X = df_with_features.copy()

        # Make predictions
        try:
            predictions = model.predict(X)

            # Create results DataFrame
            results = pd.DataFrame({"date": X.index, "close": X["close"], "predicted": predictions})

            # Calculate forecast horizons
            forecast_dates = []
            for i in range(horizon):
                next_date = X.index[-1] + timedelta(days=i + 1)
                forecast_dates.append(next_date)

            # LSTM-PPO model - get trading signals
            if "lstm_ppo" in model.models:
                lstm_ppo = model.models["lstm_ppo"]
                trading_signals = lstm_ppo.predict(X)

                if len(trading_signals) > 0:
                    # Append trading signals to results
                    signal_map = {-1.0: "Strong Sell", -0.5: "Sell", 0.0: "Hold", 0.5: "Buy", 1.0: "Strong Buy"}

                    # Map continuous values to discrete signals
                    def map_signal(value):
                        thresholds = sorted(signal_map.keys())
                        for i, threshold in enumerate(thresholds):
                            if i < len(thresholds) - 1:
                                if value >= threshold and value < thresholds[i + 1]:
                                    return signal_map[threshold]
                        return signal_map[thresholds[-1]]

                    results["signal"] = [map_signal(s) for s in trading_signals]

                    # Forecast trading signals
                    forecast_signals = []
                    for i in range(horizon):
                        # For demonstration, use the last signal
                        # In production, consider more sophisticated forecasting
                        forecast_signals.append(trading_signals[-1])

            # Create forecast DataFrame
            forecast = pd.DataFrame({"date": forecast_dates, "close": np.nan, "predicted": np.nan})

            # Generate forecast
            last_price = X["close"].iloc[-1]
            forecast_prices = []

            # Simple forecasting - in production would use ensemble model directly
            for i in range(horizon):
                if i == 0:
                    # First day forecast
                    next_price = model.predict(X.iloc[[-1]])
                    if len(next_price) > 0:
                        forecast_prices.append(next_price[0])
                    else:
                        forecast_prices.append(last_price)
                else:
                    # Create synthetic features for future dates
                    # This is simplified - in production would be more sophisticated
                    next_price = forecast_prices[-1] * (1 + np.random.normal(0, 0.005))
                    forecast_prices.append(next_price)

            forecast["predicted"] = forecast_prices

            # Combine results and forecast
            combined = pd.concat([results, forecast])

            # Store predictions
            all_predictions[symbol] = combined

            # Plot if requested
            if plot:
                plt.figure(figsize=(12, 6))
                plt.plot(results["date"], results["close"], label="Actual")
                plt.plot(combined["date"], combined["predicted"], label="Predicted", linestyle="--")

                # Mark forecast section
                plt.axvline(x=results["date"].iloc[-1], color="r", linestyle="-", alpha=0.3)
                plt.fill_between(
                    forecast["date"], forecast["predicted"] * 0.95, forecast["predicted"] * 1.05, color="r", alpha=0.2
                )

                plt.title(f"{symbol} Price Prediction (Horizon: {horizon} days)")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)

                # Save plot
                plt_path = RESULTS_DIR / f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plt_path)
                plt.close()

            # Print summary
            logger.info(f"Prediction summary for {symbol}:")
            logger.info(f"Current price: ${last_price:.2f}")
            logger.info(f"Predicted prices (next {horizon} days): {[f'${p:.2f}' for p in forecast_prices]}")

            if "lstm_ppo" in model.models and len(trading_signals) > 0:
                logger.info(f"Current trading signal: {results['signal'].iloc[-1]}")

        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")

    return all_predictions


def main():
    """Main function."""
    args = parse_args()

    # Parse symbols
    symbols = args.symbols.split(",")

    # Make predictions
    predictions = make_predictions(
        symbols=symbols,
        timeframe=args.timeframe,
        horizon=args.horizon,
        model_path=args.model_path,
        use_sentiment=args.use_sentiment,
        days=args.days,
        plot=args.plot,
    )

    # Save predictions to CSV
    for symbol, df in predictions.items():
        csv_path = RESULTS_DIR / f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions for {symbol} to {csv_path}")


if __name__ == "__main__":
    main()
