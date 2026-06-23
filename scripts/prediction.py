#!/usr/bin/env python
"""Script for making predictions with the time series ensemble model."""

import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from scripts.training import parse_model_names
from src.config import MODELS_DIR, RESULTS_DIR
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.logging_utils import configure_logging
from src.models.ensemble import EnsembleModel
from src.models.lstm_ppo import LSTMPPO

# Logging configured in main() via configure_logging() (honors --verbose and
# the per-run log file). Module-level logger is the fallback name.
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with the time series ensemble model.")

    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated list of symbols to predict")

    parser.add_argument("--timeframe", type=str, default="1d", help="Time interval (e.g., 1d, 1h)")

    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")

    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model (default: latest model)")

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

    # Ensure the loaded object is an EnsembleModel instance
    if not isinstance(model, EnsembleModel):
        # If loading the raw state dict from pickle
        if isinstance(model, dict) and "target_column" in model:
            logger.info(f"Loading ensemble state from dictionary saved at {path}")
            horizon_loaded = model.get("horizon", horizon)  # Use loaded horizon if available
            loaded_model = EnsembleModel(target_column=model["target_column"], horizon=horizon_loaded)
            # Manually restore state (similar to load method)
            loaded_model.config = model.get("config")
            loaded_model.weights = model.get("weights", {})
            loaded_model.errors = model.get("errors", {})
            loaded_model.is_fitted = model.get("is_fitted", False)
            loaded_model.models = {}  # Initialize models

            # Load models from the state dict
            loaded_model_states = model.get("models", {})
            for name, saved_model_info in loaded_model_states.items():
                if name == "lstm_ppo":
                    try:
                        lstm_ppo_path = Path(str(saved_model_info))
                        if lstm_ppo_path.exists() and (lstm_ppo_path / "model.zip").exists():
                            lstm_ppo = LSTMPPO(target_column=loaded_model.target_column, horizon=loaded_model.horizon)
                            lstm_ppo.load(lstm_ppo_path / "model.zip")
                            loaded_model.models[name] = lstm_ppo
                            loaded_model.lstm_ppo_save_path = lstm_ppo_path  # Store path
                        else:
                            logger.warning(f"LSTMPPO path/file not found during dict load: {lstm_ppo_path}")
                            loaded_model.is_fitted = False
                    except Exception as e:
                        logger.error(f"Error loading LSTMPPO from dict state: {e}")
                        loaded_model.is_fitted = False
                else:
                    # Assume other models are pickleable instances
                    loaded_model.models[name] = saved_model_info
            model = loaded_model  # Replace the dict with the constructed instance
        else:
            logger.error(f"Loaded object from {path} is not an EnsembleModel or a recognized state dictionary.")
            raise TypeError(f"Expected EnsembleModel or state dict, got {type(model)}")

    if isinstance(model, EnsembleModel):
        parse_model_names(",".join(model.models.keys()))
    return model


def make_predictions(symbols, timeframe, horizon, model_path, days, plot):
    """Make predictions for the given symbols.

    Args:
        symbols: List of symbols to predict
        timeframe: Time interval
        horizon: Forecast horizon
        model_path: Path to the saved model
        days: Number of days of historical data to use
        plot: Whether to plot predictions
    """
    # Load model
    model = load_model(model_path, horizon)

    # Initialize data loader
    data_loader = DataLoader()

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

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

        # Prepare input features
        X = df_with_features.copy()

        # Make predictions. After Phase 1.2, ensemble.predict returns positions
        # in [-1, 1] rather than predicted prices.
        try:
            predictions = model.predict(X)

            results = pd.DataFrame({"date": X.index, "close": X["close"], "position": predictions})

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

            # Phase 1.4 (B4 resolution): the prior random-walk extrapolation was
            # nonsense for prices and stayed nonsense after Phase 1.2 mapped
            # ensemble outputs into position space. An honest multi-step
            # position forecast requires simulating future OHLCV trajectories
            # (positions are derived from features that are derived from prices)
            # — out of scope for this prediction script. predictions[-1] is the
            # model's recommendation given the latest available bar; hold it
            # flat across the horizon as the baseline. Calling predict() on a
            # single-row slice is also avoided because sequence members (LSTM,
            # RL window envs) require sequence_length rows and silently return
            # empty otherwise.
            last_position = float(predictions[-1]) if len(predictions) else 0.0
            forecast_positions = [last_position] * horizon

            forecast = pd.DataFrame(
                {
                    "date": forecast_dates,
                    "close": np.nan,
                    "position": np.clip(forecast_positions, -1.0, 1.0),
                }
            )

            # Combine results and forecast
            combined = pd.concat([results, forecast])

            # Store predictions
            all_predictions[symbol] = combined

            if plot:
                fig, (ax_price, ax_pos) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                ax_price.plot(results["date"], results["close"], label="Close")
                ax_price.set_ylabel("Price")
                ax_price.grid(True)
                ax_price.legend()

                ax_pos.plot(combined["date"], combined["position"], label="Ensemble position", linestyle="--")
                ax_pos.axhline(0, color="grey", linewidth=0.5)
                ax_pos.axvline(x=results["date"].iloc[-1], color="r", linestyle="-", alpha=0.3)
                ax_pos.set_ylabel("Position")
                ax_pos.set_xlabel("Date")
                ax_pos.set_ylim(-1.05, 1.05)
                ax_pos.grid(True)
                ax_pos.legend()
                fig.suptitle(f"{symbol} — position forecast (horizon: {horizon} bars)")

                plt_path = RESULTS_DIR / f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                fig.savefig(plt_path)
                plt.close(fig)

            last_close = float(X["close"].iloc[-1])
            logger.info(f"Prediction summary for {symbol}:")
            logger.info(f"Current price: ${last_close:.2f}")
            logger.info(f"Forecast positions (next {horizon} bars): {[f'{p:+.3f}' for p in forecast_positions]}")

            if "lstm_ppo" in model.models and len(trading_signals) > 0:
                logger.info(f"Current trading signal: {results['signal'].iloc[-1]}")

        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")

    return all_predictions


def main():
    """Main function."""
    args = parse_args()
    configure_logging(verbose=args.verbose, run_name="prediction")

    # Parse symbols
    symbols = args.symbols.split(",")

    # Make predictions
    predictions = make_predictions(
        symbols=symbols,
        timeframe=args.timeframe,
        horizon=args.horizon,
        model_path=args.model_path,
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
