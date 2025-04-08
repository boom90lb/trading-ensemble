# src/features.py
"""Feature engineering for the time series ensemble model."""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for time series data."""

    def __init__(
        self,
        price_scaler: str = "minmax",
        volume_scaler: str = "minmax",
        technical_scaler: str = "standard",
        sentiment_scaler: str = "standard",
    ):
        """Initialize the feature engineer.

        Args:
            price_scaler: Scaler for price features ('minmax', 'standard', or 'none')
            volume_scaler: Scaler for volume features ('minmax', 'standard', or 'none')
            technical_scaler: Scaler for technical indicators ('minmax', 'standard', or 'none')
            sentiment_scaler: Scaler for sentiment features ('minmax', 'standard', or 'none')
        """
        self.price_scaler = price_scaler
        self.volume_scaler = volume_scaler
        self.technical_scaler = technical_scaler
        self.sentiment_scaler = sentiment_scaler

        # Dictionaries to store scalers for each symbol
        self.price_scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}
        self.volume_scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}
        self.technical_scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}
        self.sentiment_scalers: Dict[str, Union[MinMaxScaler, StandardScaler]] = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Check required columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return result

        try:
            # 1. Price-based features
            # Moving averages
            result["ma5"] = result["close"].rolling(window=5).mean()
            result["ma10"] = result["close"].rolling(window=10).mean()
            result["ma20"] = result["close"].rolling(window=20).mean()
            result["ma50"] = result["close"].rolling(window=50).mean()

            # Exponential moving averages
            result["ema5"] = result["close"].ewm(span=5, adjust=False).mean()
            result["ema10"] = result["close"].ewm(span=10, adjust=False).mean()
            result["ema20"] = result["close"].ewm(span=20, adjust=False).mean()

            # Price changes
            result["price_change"] = result["close"].pct_change()
            result["price_change_5"] = result["close"].pct_change(periods=5)
            result["price_change_10"] = result["close"].pct_change(periods=10)

            # Volatility
            result["volatility_5"] = result["close"].rolling(window=5).std()
            result["volatility_10"] = result["close"].rolling(window=10).std()
            result["volatility_20"] = result["close"].rolling(window=20).std()

            # 2. Volume-based features (if available)
            if "volume" in result.columns:
                # Volume changes
                result["volume_change"] = result["volume"].pct_change()
                result["volume_ma5"] = result["volume"].rolling(window=5).mean()
                result["volume_ma10"] = result["volume"].rolling(window=10).mean()

                # Price-volume relationship
                result["price_volume_ratio"] = result["close"] / (
                    result["volume"] + 1
                )  # Add 1 to avoid division by zero

            # 3. Technical indicators
            # Relative Strength Index (RSI)
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0)  # type: ignore
            loss = -delta.where(delta < 0, 0)  # type: ignore

            avg_gain_14 = gain.rolling(window=14).mean()
            avg_loss_14 = loss.rolling(window=14).mean()

            rs_14 = avg_gain_14 / (avg_loss_14 + 1e-10)  # Add small value to avoid division by zero
            result["rsi_14"] = 100 - (100 / (1 + rs_14))

            # Moving Average Convergence Divergence (MACD)
            result["macd"] = (
                result["close"].ewm(span=12, adjust=False).mean() - result["close"].ewm(span=26, adjust=False).mean()
            )
            result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
            result["macd_hist"] = result["macd"] - result["macd_signal"]

            # Bollinger Bands
            result["bb_middle"] = result["close"].rolling(window=20).mean()
            bb_std = result["close"].rolling(window=20).std()
            result["bb_upper"] = result["bb_middle"] + (bb_std * 2)
            result["bb_lower"] = result["bb_middle"] - (bb_std * 2)
            result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

            # 4. Price direction features
            # Previous close to current close
            result["direction_1"] = np.where(result["close"].diff() > 0, 1, -1)  # type: ignore

            # Short-term trend (5 days)
            result["direction_5"] = np.where(result["close"].diff(5) > 0, 1, -1)  # type: ignore

            # Medium-term trend (10 days)
            result["direction_10"] = np.where(result["close"].diff(10) > 0, 1, -1)  # type: ignore

            # 5. Gap features
            result["gap"] = (result["open"] - result["close"].shift(1)) / result["close"].shift(1)

            # 6. Range features
            result["daily_range"] = (result["high"] - result["low"]) / result["close"]
            result["daily_range_ma5"] = result["daily_range"].rolling(window=5).mean()

            # 7. Pattern detection (simplified)
            # Doji pattern (open close very close, long range)
            result["doji"] = (
                (abs(result["open"] - result["close"]) / (result["high"] - result["low"] + 1e-10) < 0.1)
                & ((result["high"] - result["low"]) / result["low"] > 0.01)
            ).astype(int)

        except Exception as e:
            logger.error(f"Error creating features: {e}")

        # Drop rows with NaN values if needed
        # result = result.dropna()

        return result

    def create_lagged_features(self, df: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
        """Create lagged features for time series prediction.

        Args:
            df: DataFrame with features
            lag_periods: List of periods to lag

        Returns:
            DataFrame with lagged features
        """
        result = df.copy()

        try:
            # Get all numeric columns except target columns (which start with "target_")
            numeric_cols = [
                col
                for col in df.select_dtypes(include=["number"]).columns
                if not (col.startswith("target_") or col.startswith("direction_"))
            ]

            # Create lagged features
            for period in lag_periods:
                if period <= 0:
                    continue

                for col in numeric_cols:
                    result[f"{col}_lag{period}"] = result[col].shift(period)

        except Exception as e:
            logger.error(f"Error creating lagged features: {e}")

        return result

    def create_target_variable(self, df: pd.DataFrame, price_col: str = "close", horizon: int = 5) -> pd.DataFrame:
        """Create target variable for forecasting.

        Args:
            df: DataFrame with price data
            price_col: Column name for price data
            horizon: Forecast horizon

        Returns:
            DataFrame with target variable
        """
        result = df.copy()

        try:
            # Create future price target
            result[f"target_{horizon}"] = result[price_col].shift(-horizon)

            # Create direction target (1 for up, -1 for down)
            result[f"direction_{horizon}"] = np.where(result[f"target_{horizon}"] > result[price_col], 1, -1)

        except Exception as e:
            logger.error(f"Error creating target variable: {e}")

        return result

    def fit_scalers(self, df: pd.DataFrame, symbol: str) -> None:
        """Fit scalers on training data.

        Args:
            df: Training DataFrame
            symbol: Symbol for the data
        """
        try:
            # Get column groups
            price_cols = [
                col
                for col in df.columns
                if col.lower() in ["open", "high", "low", "close"] or col.startswith("ma") or col.startswith("ema")
            ]

            volume_cols = [col for col in df.columns if "volume" in col.lower()]

            technical_cols = [
                col
                for col in df.select_dtypes(include=["number"]).columns
                if col not in price_cols
                and col not in volume_cols
                and not col.startswith("target_")
                and not col.startswith("direction_")
            ]

            sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]

            # Initialize and fit price scaler
            if self.price_scaler.lower() != "none" and price_cols:
                if self.price_scaler.lower() == "minmax":
                    self.price_scalers[symbol] = MinMaxScaler()
                else:
                    self.price_scalers[symbol] = StandardScaler()

                self.price_scalers[symbol].fit(df[price_cols].dropna())

            # Initialize and fit volume scaler
            if self.volume_scaler.lower() != "none" and volume_cols:
                if self.volume_scaler.lower() == "minmax":
                    self.volume_scalers[symbol] = MinMaxScaler()
                else:
                    self.volume_scalers[symbol] = StandardScaler()

                self.volume_scalers[symbol].fit(df[volume_cols].dropna())

            # Initialize and fit technical scaler
            if self.technical_scaler.lower() != "none" and technical_cols:
                if self.technical_scaler.lower() == "minmax":
                    self.technical_scalers[symbol] = MinMaxScaler()
                else:
                    self.technical_scalers[symbol] = StandardScaler()

                self.technical_scalers[symbol].fit(df[technical_cols].dropna())

            # Initialize and fit sentiment scaler
            if self.sentiment_scaler.lower() != "none" and sentiment_cols:
                if self.sentiment_scaler.lower() == "minmax":
                    self.sentiment_scalers[symbol] = MinMaxScaler()
                else:
                    self.sentiment_scalers[symbol] = StandardScaler()

                self.sentiment_scalers[symbol].fit(df[sentiment_cols].dropna())

        except Exception as e:
            logger.error(f"Error fitting scalers for {symbol}: {e}")

    def transform_features(self, df: pd.DataFrame, symbol: str, is_train: bool = False) -> pd.DataFrame:
        """Transform features using fitted scalers.

        Args:
            df: DataFrame with features
            symbol: Symbol for the data
            is_train: Whether this is training data

        Returns:
            DataFrame with transformed features
        """
        result = df.copy()

        try:
            # Get column groups
            price_cols = [
                col
                for col in df.columns
                if col.lower() in ["open", "high", "low", "close"] or col.startswith("ma") or col.startswith("ema")
            ]

            volume_cols = [col for col in df.columns if "volume" in col.lower()]

            technical_cols = [
                col
                for col in df.select_dtypes(include=["number"]).columns
                if col not in price_cols
                and col not in volume_cols
                and not col.startswith("target_")
                and not col.startswith("direction_")
            ]

            sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]

            # Transform price features
            if symbol in self.price_scalers and price_cols:
                price_scaler = self.price_scalers[symbol]
                result[price_cols] = result[price_cols].fillna(method="ffill")  # type: ignore
                # Check if data exists before transforming
                if not result[price_cols].empty:
                    transformed_data = price_scaler.transform(result[price_cols])
                    result[price_cols] = transformed_data  # Assign numpy array directly

            # Transform volume features
            if symbol in self.volume_scalers and volume_cols:
                volume_scaler = self.volume_scalers[symbol]
                result[volume_cols] = result[volume_cols].fillna(method="ffill")  # type: ignore
                # Check if data exists before transforming
                if not result[volume_cols].empty:
                    transformed_data = volume_scaler.transform(result[volume_cols])
                    result[volume_cols] = transformed_data  # Assign numpy array directly

            # Transform technical features
            if symbol in self.technical_scalers and technical_cols:
                technical_scaler = self.technical_scalers[symbol]
                result[technical_cols] = result[technical_cols].fillna(method="ffill")  # type: ignore
                # Check if data exists before transforming
                if not result[technical_cols].empty:
                    transformed_data = technical_scaler.transform(result[technical_cols])
                    result[technical_cols] = transformed_data  # Assign numpy array directly

            # Transform sentiment features
            if symbol in self.sentiment_scalers and sentiment_cols:
                sentiment_scaler = self.sentiment_scalers[symbol]
                result[sentiment_cols] = result[sentiment_cols].fillna(0)  # Fill sentiment NaNs with 0
                # Check if data exists before transforming
                if not result[sentiment_cols].empty:
                    transformed_data = sentiment_scaler.transform(result[sentiment_cols])
                    result[sentiment_cols] = transformed_data  # Assign numpy array directly

            # Handle remaining NaNs
            result = result.fillna(method="ffill").fillna(0)  # type: ignore

        except Exception as e:
            logger.error(f"Error transforming features for {symbol}: {e}")

        return result
