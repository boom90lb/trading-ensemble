# src/trading.py
"""Trading strategy and backtesting module."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from src.config import RESULTS_DIR, TradingConfig
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)


class Position(Enum):
    """Enum for position types."""

    LONG = 1
    SHORT = -1
    FLAT = 0


# Define a TypedDict for clearer position information structure
class PositionInfo(TypedDict):
    position: Position
    entry_price: float
    size: float


class TradingStrategy:
    """Trading strategy using ensemble model predictions and sentiment analysis."""

    def __init__(
        self,
        model: BaseModel,
        config: TradingConfig,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        use_sentiment: bool = True,
    ):
        """Initialize the trading strategy.

        Args:
            model: Trained forecasting model
            config: Trading configuration
            sentiment_analyzer: Sentiment analyzer for news data
            use_sentiment: Whether to use sentiment data in trading decisions
        """
        self.model = model
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.use_sentiment = use_sentiment and sentiment_analyzer is not None
        self.positions: Dict[str, PositionInfo] = {}
        self.cash = config.initial_capital
        self.portfolio_value = config.initial_capital
        self.history: List[Dict[str, Union[pd.Timestamp, float]]] = []
        self.transaction_log: List[Dict[str, Any]] = []

    def calculate_signal(
        self,
        predictions: np.ndarray,
        current_price: float,
        symbol: str,
        date: pd.Timestamp,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[Position, float]:
        """Calculate trading signal based on model predictions and sentiment.

        Args:
            predictions: Model predictions
            current_price: Current price
            symbol: Symbol
            date: Current date
            sentiment_data: Sentiment data for the current date

        Returns:
            Tuple of (position, confidence)
        """
        # Default confidence
        confidence = 0.5

        # Get prediction for the current step
        if len(predictions) > 0:
            predicted_price = predictions[0]
            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price

            # Adjust confidence based on ensemble model if available
            if isinstance(self.model, EnsembleModel):
                # Get model contributions
                try:
                    contributions = self.model.get_model_contributions(pd.DataFrame(index=[date]))
                    if not contributions.empty:
                        # Calculate confidence based on agreement between models
                        model_preds = [contributions[f"{name}_pred"].iloc[0] for name in self.model.models.keys()]
                        signs = [np.sign(pred - current_price) for pred in model_preds if not np.isnan(pred)]
                        if signs:
                            agreement = np.abs(np.mean(signs))
                            # Scale confidence based on agreement (0.5-1.0)
                            confidence = 0.5 + (agreement * 0.5)
                except Exception as e:
                    logger.error(f"Error calculating model contributions: {e}")
        else:
            logger.warning("No predictions available for signal calculation")
            return Position.FLAT, confidence

        # Incorporate sentiment if available
        if self.use_sentiment and sentiment_data is not None:
            try:
                # Get sentiment score and momentum
                sentiment_cols = [col for col in sentiment_data.columns if col.startswith(f"{symbol}_sentiment_")]

                if sentiment_cols:
                    # Get sentiment score
                    sentiment_score_col = f"{symbol}_sentiment_score"
                    if sentiment_score_col in sentiment_data.columns:
                        sentiment_score = sentiment_data[sentiment_score_col].iloc[0]
                    else:
                        sentiment_score = 0

                    # Get sentiment momentum
                    sentiment_momentum_col = f"{symbol}_sentiment_momentum"
                    if sentiment_momentum_col in sentiment_data.columns:
                        sentiment_momentum = sentiment_data[sentiment_momentum_col].iloc[0]
                    else:
                        sentiment_momentum = 0

                    # Adjust predicted return based on sentiment
                    sentiment_factor = 0.3  # Weight for sentiment adjustment
                    adjusted_return = (1 - sentiment_factor) * predicted_return + sentiment_factor * (
                        sentiment_score + sentiment_momentum
                    ) / 3.0

                    # Adjust confidence based on sentiment strength
                    sentiment_strength = abs(sentiment_score) + abs(sentiment_momentum)
                    confidence += min(0.2, sentiment_strength / 10.0)  # Max 0.2 boost

                    # Update predicted return
                    predicted_return = adjusted_return
            except Exception as e:
                logger.error(f"Error incorporating sentiment data: {e}")

        # Determine position based on predicted return
        # Add a threshold to avoid overtrading on small signals
        threshold = 0.002  # 0.2% minimum expected return

        if predicted_return > threshold:
            return Position.LONG, min(1.0, confidence)
        elif predicted_return < -threshold:
            return Position.SHORT, min(1.0, confidence)
        else:
            return Position.FLAT, confidence

    def execute_trade(
        self, symbol: str, position: Position, confidence: float, price: float, date: pd.Timestamp
    ) -> None:
        """Execute a trade based on the signal.

        Args:
            symbol: Symbol to trade
            position: Target position
            confidence: Confidence in the signal (0-1)
            price: Current price
            date: Trade date
        """
        current_position = self.positions.get(symbol, {"position": Position.FLAT, "entry_price": 0, "size": 0})

        # If position is the same, do nothing
        if current_position["position"] == position:
            return

        # Calculate position size based on confidence and max position size
        position_size = self.config.position_size * confidence

        # Calculate number of shares based on current portfolio value
        max_allocation = self.portfolio_value * position_size
        shares = max_allocation / price

        # Close existing position if any
        if current_position["position"] != Position.FLAT:
            # Calculate profit/loss
            old_position_type = current_position["position"]
            old_entry_price = current_position["entry_price"]
            old_size = current_position["size"]

            # Calculate profit/loss (accounting for short positions)
            if old_position_type == Position.LONG:
                pnl = (price - old_entry_price) * old_size  # type: ignore
            else:  # SHORT
                pnl = (old_entry_price - price) * old_size  # type: ignore

            # Deduct commission
            commission = price * old_size * self.config.commission  # type: ignore
            net_pnl = pnl - commission

            # Update cash
            self.cash += max_allocation + net_pnl

            # Log transaction
            self.transaction_log.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": f"CLOSE_{old_position_type.name}",
                    "price": price,
                    "shares": old_size,
                    "commission": commission,
                    "pnl": pnl,
                    "net_pnl": net_pnl,
                    "cash": self.cash,
                    "portfolio_value": self.portfolio_value,
                }
            )

            logger.info(
                f"Closed {old_position_type.name} position on {symbol} at {price:.2f}, "
                f"PnL: {net_pnl:.2f}, Cash: {self.cash:.2f}"
            )

        # Open new position if not FLAT
        if position != Position.FLAT:
            # Calculate commission
            commission = price * shares * self.config.commission

            # Update cash
            self.cash -= max_allocation + commission

            # Update positions
            self.positions[symbol] = {"position": position, "entry_price": price, "size": shares}

            # Log transaction
            self.transaction_log.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": f"OPEN_{position.name}",
                    "price": price,
                    "shares": shares,
                    "commission": commission,
                    "pnl": 0,
                    "net_pnl": -commission,
                    "cash": self.cash,
                    "portfolio_value": self.portfolio_value,
                }
            )

            logger.info(
                f"Opened {position.name} position on {symbol} at {price:.2f}, "
                f"Shares: {shares:.2f}, Cash: {self.cash:.2f}"
            )
        else:
            # If new position is FLAT, just update the positions dictionary
            self.positions[symbol] = {"position": Position.FLAT, "entry_price": 0, "size": 0}

    def update_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Update portfolio value based on current positions and prices.

        Args:
            prices: Dictionary mapping symbols to current prices

        Returns:
            Updated portfolio value
        """
        position_value = 0.0  # Initialize as float

        # Calculate value of open positions
        for symbol, position_info in self.positions.items():
            if position_info["position"] != Position.FLAT and symbol in prices:
                current_price = prices[symbol]
                size = position_info["size"]
                entry_price = position_info["entry_price"]

                if position_info["position"] == Position.LONG:
                    value = size * current_price  # type: ignore
                else:  # SHORT
                    value = size * (2 * entry_price - current_price)  # type: ignore

                position_value += value

        # Update portfolio value
        self.portfolio_value = self.cash + position_value

        return float(self.portfolio_value)

    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sentiment: bool = True,
    ) -> pd.DataFrame:
        """Run a backtest of the trading strategy.

        Args:
            data: Dictionary mapping symbols to DataFrames with price data
            start_date: Start date for backtest (default: use all data)
            end_date: End date for backtest (default: use all data)
            use_sentiment: Whether to use sentiment data in trading decisions

        Returns:
            DataFrame with backtest results
        """
        # Reset state
        self.positions = {}
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.history = []
        self.transaction_log = []

        # Set date range
        if len(data) == 0:
            logger.warning("No data provided for backtesting")
            return pd.DataFrame()

        # Get a list of all dates across all symbols
        all_dates = pd.DatetimeIndex([])
        for symbol, df in data.items():
            all_dates = all_dates.union(df.index)

        # Ensure it's a DatetimeIndex after union
        all_dates = pd.DatetimeIndex(all_dates).sort_values()

        # Filter by date range if specified
        if start_date:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]

        if len(all_dates) == 0:
            logger.warning("No dates in the specified range")
            return pd.DataFrame()

        logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")

        # Get sentiment data if requested
        sentiment_data = {}
        if use_sentiment and self.use_sentiment and self.sentiment_analyzer:
            for symbol in data.keys():
                try:
                    sentiment_features = self.sentiment_analyzer.create_sentiment_features(symbol, all_dates)
                    sentiment_data[symbol] = sentiment_features
                except Exception as e:
                    logger.error(f"Error creating sentiment features for {symbol}: {e}")
                    sentiment_data[symbol] = pd.DataFrame(index=all_dates)

        # Run backtest
        for date in all_dates:
            # Get current prices for all symbols
            current_prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    # Ensure price is float, ignore type checker uncertainty
                    current_prices[symbol] = float(df.loc[date, "close"])  # type: ignore

            # Skip if no prices available
            if not current_prices:
                continue

            # Update portfolio value
            self.update_portfolio_value(current_prices)

            # For each symbol with data for this date
            for symbol, price in current_prices.items():
                # Get data for prediction
                symbol_data = data[symbol]
                if date not in symbol_data.index:
                    continue

                # Create features DataFrame for the current date
                features_df = symbol_data.loc[:date].iloc[-1:].copy()

                # Add sentiment features if available
                symbol_sentiment = None
                if use_sentiment and symbol in sentiment_data:
                    # Use slice indexing for .loc to ensure DataFrame return type
                    symbol_sentiment = sentiment_data[symbol].loc[date:date].copy()

                # Generate prediction
                try:
                    predictions = self.model.predict(features_df)

                    # Calculate signal
                    position, confidence = self.calculate_signal(predictions, price, symbol, date, symbol_sentiment)

                    # Execute trade
                    self.execute_trade(symbol, position, confidence, price, date)

                except Exception as e:
                    logger.error(f"Error during backtest for {symbol} on {date}: {e}")
                    continue

            # Update portfolio value after all trades
            portfolio_value = self.update_portfolio_value(current_prices)

            # Record history
            self.history.append({"date": date, "portfolio_value": portfolio_value, "cash": self.cash})

        # Create results DataFrame
        results = pd.DataFrame(self.history)
        if not results.empty:
            results = results.set_index("date")

            # Calculate returns
            results["daily_return"] = results["portfolio_value"].pct_change()
            results["cumulative_return"] = (1 + results["daily_return"]).cumprod() - 1

            # Calculate metrics
            results["drawdown"] = 1 - results["portfolio_value"] / results["portfolio_value"].cummax()

        return results

    def calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from backtest results.

        Args:
            results: DataFrame with backtest results

        Returns:
            Dictionary with performance metrics
        """
        if results.empty:
            return {}

        # Basic metrics
        metrics = {}

        # Total return
        metrics["total_return"] = results["cumulative_return"].iloc[-1]

        # Annualized return
        days = (results.index[-1] - results.index[0]).days
        if days > 0:
            metrics["annualized_return"] = (1 + metrics["total_return"]) ** (365 / days) - 1

        # Volatility
        metrics["volatility"] = results["daily_return"].std() * np.sqrt(252)

        # Sharpe ratio
        if metrics["volatility"] > 0:
            excess_return = metrics["annualized_return"] - self.config.risk_free_rate
            metrics["sharpe_ratio"] = excess_return / metrics["volatility"]

        # Maximum drawdown
        metrics["max_drawdown"] = results["drawdown"].max()

        # Calmar ratio
        if metrics["max_drawdown"] > 0:
            metrics["calmar_ratio"] = metrics["annualized_return"] / metrics["max_drawdown"]

        # Win rate from transaction log
        if self.transaction_log:
            trades_df = pd.DataFrame(self.transaction_log)
            close_trades = trades_df[trades_df["action"].str.startswith("CLOSE")]

            if len(close_trades) > 0:
                winning_trades = close_trades[close_trades["pnl"] > 0]
                metrics["win_rate"] = len(winning_trades) / len(close_trades)

                # Average profit/loss
                metrics["avg_profit"] = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
                losing_trades = close_trades[close_trades["pnl"] <= 0]
                metrics["avg_loss"] = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0

                # Profit factor
                total_profit = winning_trades["pnl"].sum()
                total_loss = abs(losing_trades["pnl"].sum())
                if total_loss > 0:
                    metrics["profit_factor"] = total_profit / total_loss

        return metrics

    def plot_results(self, results: pd.DataFrame, show_trades: bool = True) -> None:
        """Plot backtest results.

        Args:
            results: DataFrame with backtest results
            show_trades: Whether to show individual trades on the plot
        """
        if results.empty:
            logger.warning("No results to plot")
            return

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        # Plot portfolio value
        results["portfolio_value"].plot(ax=ax1, label="Portfolio Value", color="blue")
        ax1.set_ylabel("Portfolio Value")
        ax1.set_title("Backtest Results")
        ax1.legend()
        ax1.grid(True)

        # Format date axis
        date_format = DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_format)

        # Plot drawdown
        results["drawdown"].plot(ax=ax2, label="Drawdown", color="red", fill=True, alpha=0.3)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, max(0.01, results["drawdown"].max() * 1.1))  # Ensure some vertical space
        ax2.grid(True)

        # Plot trades if requested
        if show_trades and self.transaction_log:
            trades_df = pd.DataFrame(self.transaction_log)

            # Plot buy signals
            buy_trades = trades_df[trades_df["action"] == "OPEN_LONG"]
            if not buy_trades.empty:
                ax1.scatter(
                    buy_trades["date"], buy_trades["portfolio_value"], marker="^", color="green", s=100, label="Buy"
                )

            # Plot sell signals
            sell_trades = trades_df[trades_df["action"] == "CLOSE_LONG"]
            if not sell_trades.empty:
                ax1.scatter(
                    sell_trades["date"], sell_trades["portfolio_value"], marker="v", color="red", s=100, label="Sell"
                )

            # Plot short signals
            short_trades = trades_df[trades_df["action"] == "OPEN_SHORT"]
            if not short_trades.empty:
                ax1.scatter(
                    short_trades["date"],
                    short_trades["portfolio_value"],
                    marker="v",
                    color="purple",
                    s=100,
                    label="Short",
                )

            # Plot cover signals
            cover_trades = trades_df[trades_df["action"] == "CLOSE_SHORT"]
            if not cover_trades.empty:
                ax1.scatter(
                    cover_trades["date"],
                    cover_trades["portfolio_value"],
                    marker="^",
                    color="orange",
                    s=100,
                    label="Cover",
                )

            # Update legend
            if not all(df.empty for df in [buy_trades, sell_trades, short_trades, cover_trades]):
                ax1.legend()

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = RESULTS_DIR / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fig_path)
        logger.info(f"Results plot saved to {fig_path}")

        # Show figure
        plt.show()

    def generate_report(self, results: pd.DataFrame) -> str:
        """Generate a backtest report.

        Args:
            results: DataFrame with backtest results

        Returns:
            Report as a string
        """
        if results.empty:
            return "No results to report"

        metrics = self.calculate_performance_metrics(results)

        report = []
        report.append("=" * 50)
        report.append("BACKTEST REPORT")
        report.append("=" * 50)
        report.append("")

        # Period
        report.append(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
        report.append(f"Trading days: {len(results)}")
        report.append("")

        # Portfolio performance
        report.append("-" * 50)
        report.append("PORTFOLIO PERFORMANCE")
        report.append("-" * 50)
        report.append(f"Initial capital: ${self.config.initial_capital:.2f}")
        report.append(f"Final portfolio value: ${results['portfolio_value'].iloc[-1]:.2f}")
        report.append(f"Total return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"Volatility (annualized): {metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Maximum drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Calmar ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report.append("")

        # Trading statistics
        report.append("-" * 50)
        report.append("TRADING STATISTICS")
        report.append("-" * 50)

        if self.transaction_log:
            trades_df = pd.DataFrame(self.transaction_log)
            opens = trades_df[trades_df["action"].str.startswith("OPEN")]
            closes = trades_df[trades_df["action"].str.startswith("CLOSE")]

            report.append(f"Total trades: {len(opens)}")

            if "win_rate" in metrics:
                report.append(f"Win rate: {metrics['win_rate']:.2%}")
                report.append(f"Average profit: ${metrics.get('avg_profit', 0):.2f}")
                report.append(f"Average loss: ${metrics.get('avg_loss', 0):.2f}")
                report.append(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")

            # Calculate by position type
            long_opens = trades_df[trades_df["action"] == "OPEN_LONG"]
            long_closes = trades_df[trades_df["action"] == "CLOSE_LONG"]
            short_opens = trades_df[trades_df["action"] == "OPEN_SHORT"]
            short_closes = trades_df[trades_df["action"] == "CLOSE_SHORT"]

            report.append(f"Long trades: {len(long_opens)}")
            report.append(f"Short trades: {len(short_opens)}")

            # Calculate total commission
            total_commission = trades_df["commission"].sum()
            report.append(f"Total commission: ${total_commission:.2f}")

        report.append("")
        report.append("=" * 50)

        return "\n".join(report)
