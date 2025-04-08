#!/usr/bin/env python
"""Script for backtesting the time series ensemble trading strategy."""

import argparse
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import MODELS_DIR, RESULTS_DIR, TradingConfig
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.models.lstm_ppo import LSTMPPO
from src.sentiment_analysis import SentimentAnalyzer
from src.trading import TradingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Backtest the time series ensemble trading strategy.")

    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated list of symbols to backtest")

    parser.add_argument("--timeframe", type=str, default="1d", help="Time interval (e.g., 1d, 1h)")

    parser.add_argument("--start_date", type=str, required=True, help="Start date for backtest in YYYY-MM-DD format")

    parser.add_argument(
        "--end_date", type=str, default=None, help="End date for backtest in YYYY-MM-DD format (default: today)"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model (default: latest model)")

    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")

    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital for backtest")

    parser.add_argument(
        "--position_size", type=float, default=0.1, help="Position size as a fraction of portfolio (0-1)"
    )

    parser.add_argument("--transaction_cost", type=float, default=0.001, help="Transaction cost ratio")

    parser.add_argument("--stop_loss", type=float, default=0.02, help="Stop loss as a fraction of entry price (0-1)")

    parser.add_argument(
        "--take_profit", type=float, default=0.05, help="Take profit as a fraction of entry price (0-1)"
    )

    parser.add_argument("--use_sentiment", action="store_true", help="Use sentiment analysis")

    parser.add_argument("--use_ppo", action="store_true", help="Use LSTM-PPO model for trading decisions")

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


def run_backtest(args):
    """Run the backtest.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with backtest results
    """
    # Parse symbols
    symbols = args.symbols.split(",")

    # Load model
    model = load_model(args.model_path, args.horizon)

    # Initialize data loader
    data_loader = DataLoader()

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Initialize sentiment analyzer if requested
    sentiment_analyzer = None
    if args.use_sentiment:
        logger.info("Initializing sentiment analyzer")
        sentiment_analyzer = SentimentAnalyzer()

    # Set end date if not provided
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    # Create trading config
    trading_config = TradingConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        commission=args.transaction_cost,
    )

    # Determine which model to use for trading decisions
    if args.use_ppo and "lstm_ppo" in model.models:
        logger.info("Using LSTM-PPO model for trading decisions")
        trading_model = model.models["lstm_ppo"]
    else:
        logger.info("Using ensemble model for trading decisions")
        trading_model = model

    # Initialize trading strategy
    strategy = TradingStrategy(
        model=trading_model,
        config=trading_config,
        sentiment_analyzer=sentiment_analyzer,
        use_sentiment=args.use_sentiment,
    )

    # Fetch and prepare data for backtesting
    backtest_data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")

        # Fetch historical data
        df = data_loader.fetch_historical_data(
            symbol=symbol, interval=args.timeframe, start_date=args.start_date, end_date=end_date
        )

        if df.empty:
            logger.warning(f"No data found for {symbol}")
            continue

        # Create features
        df_with_features = feature_engineer.create_features(df)
        df_with_features = feature_engineer.create_lagged_features(df_with_features, [1, 2, 5, 10])

        # Add sentiment features if requested
        if args.use_sentiment and sentiment_analyzer:
            logger.info(f"Creating sentiment features for {symbol}")
            sentiment_features = sentiment_analyzer.create_sentiment_features(symbol, df_with_features.index)

            if not sentiment_features.empty:
                df_with_features = df_with_features.join(sentiment_features)

        # Store prepared data
        backtest_data[symbol] = df_with_features

    if not backtest_data:
        logger.error("No data available for backtesting")
        return {}

    # Run backtest
    logger.info("Running backtest")
    results = strategy.backtest(
        data=backtest_data, start_date=args.start_date, end_date=end_date, use_sentiment=args.use_sentiment
    )

    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(results)

    # Plot results
    strategy.plot_results(results, show_trades=True)

    # Generate report
    report = strategy.generate_report(results)

    # Print report
    print("\n" + report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"backtest_results_{timestamp}.csv"
    results.to_csv(results_path)

    # Save report
    report_path = RESULTS_DIR / f"backtest_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save metrics
    metrics_path = RESULTS_DIR / f"backtest_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    logger.info(f"Results saved to {RESULTS_DIR}")

    return {"results": results, "metrics": metrics, "report": report}


def compare_models_backtest(args):
    """Run backtest with both ensemble and LSTM-PPO models for comparison.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with comparison results
    """
    # Create a copy of args for ensemble model
    ensemble_args = argparse.Namespace(**vars(args))
    ensemble_args.use_ppo = False

    # Create a copy of args for LSTM-PPO model
    ppo_args = argparse.Namespace(**vars(args))
    ppo_args.use_ppo = True

    # Run backtest with ensemble model
    logger.info("Running backtest with ensemble model")
    ensemble_results = run_backtest(ensemble_args)

    # Run backtest with LSTM-PPO model
    logger.info("Running backtest with LSTM-PPO model")
    ppo_results = run_backtest(ppo_args)

    if not ensemble_results or not ppo_results:
        logger.error("Failed to run comparison backtest")
        return {}

    # Compare metrics
    comparison = {"ensemble": ensemble_results["metrics"], "lstm_ppo": ppo_results["metrics"]}

    # Plot comparison
    plt.figure(figsize=(15, 8))

    # Portfolio value comparison
    plt.subplot(2, 1, 1)
    ensemble_results["results"]["portfolio_value"].plot(label="Ensemble Model")
    ppo_results["results"]["portfolio_value"].plot(label="LSTM-PPO Model")
    plt.title("Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)

    # Drawdown comparison
    plt.subplot(2, 1, 2)
    ensemble_results["results"]["drawdown"].plot(label="Ensemble Model", color="orange")
    ppo_results["results"]["drawdown"].plot(label="LSTM-PPO Model", color="green")
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = RESULTS_DIR / f"model_comparison_{timestamp}.png"
    plt.savefig(comparison_path)

    # Create comparison table
    comparison_table = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print(comparison_table)

    # Save comparison table
    comparison_table_path = RESULTS_DIR / f"model_comparison_{timestamp}.csv"
    comparison_table.to_csv(comparison_table_path)

    logger.info(f"Comparison results saved to {RESULTS_DIR}")

    return comparison


def main():
    """Main function."""
    args = parse_args()

    # Check if both models should be compared
    if args.use_ppo and "all" in args.symbols.lower():
        logger.info("Running comparison backtest between ensemble and LSTM-PPO models")
        # Update symbols to use specific ones instead of 'all'
        args.symbols = "AAPL,MSFT,GOOG,AMZN" if args.symbols.lower() == "all" else args.symbols
        compare_models_backtest(args)
    else:
        # Run standard backtest
        run_backtest(args)


if __name__ == "__main__":
    main()
