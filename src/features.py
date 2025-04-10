# src/features.py
"""Feature engineering for the time series ensemble model using NNX."""

import logging
from typing import Dict, List, Optional

import flax.nnx as nnx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScalerModule(nnx.Module):
    """NNX module for feature scaling."""

    def __init__(self, *, scaler_type: str = "minmax", rngs: Optional[nnx.Rngs] = None):
        """Initialize the scaler module.

        Args:
            scaler_type: Type of scaler ('minmax', 'standard', or 'none')
            rngs: Random number generators
        """
        self.scaler_type = scaler_type.lower()
        self.is_fitted = nnx.Param(False)
        self.mean = nnx.Param(np.array([0.0]))
        self.std = nnx.Param(np.array([1.0]))
        self.min_vals = nnx.Param(np.array([0.0]))
        self.max_vals = nnx.Param(np.array([1.0]))

    def fit(self, data: np.ndarray) -> None:
        """Fit the scaler to the data.

        Args:
            data: Data to fit the scaler to
        """
        if self.scaler_type == "none" or data.size == 0:
            return

        if self.scaler_type == "standard":
            self.mean.value = np.nanmean(data, axis=0)
            self.std.value = np.nanstd(data, axis=0)
            # Avoid division by zero
            self.std.value = np.where(self.std.value == 0, 1.0, self.std.value)
        elif self.scaler_type == "minmax":
            self.min_vals.value = np.nanmin(data, axis=0)
            self.max_vals.value = np.nanmax(data, axis=0)
            # Avoid division by zero
            range_vals = self.max_vals.value - self.min_vals.value
            range_vals = np.where(range_vals == 0, 1.0, range_vals)
            self.max_vals.value = self.min_vals.value + range_vals

        self.is_fitted.value = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scaler.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        if self.scaler_type == "none" or not self.is_fitted.value or data.size == 0:
            return data

        result = data.copy()

        if self.scaler_type == "standard":
            result = (result - self.mean.value) / self.std.value
        elif self.scaler_type == "minmax":
            result = (result - self.min_vals.value) / (self.max_vals.value - self.min_vals.value)

        return result


class FeatureEngineer(nnx.Module):
    """Feature engineering for time series data using NNX."""

    def __init__(
        self,
        *,
        price_scaler: str = "minmax",
        volume_scaler: str = "minmax",
        technical_scaler: str = "standard",
        sentiment_scaler: str = "standard",
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Initialize the feature engineer.

        Args:
            price_scaler: Scaler for price features ('minmax', 'standard', or 'none')
            volume_scaler: Scaler for volume features ('minmax', 'standard', or 'none')
            technical_scaler: Scaler for technical indicators ('minmax', 'standard', or 'none')
            sentiment_scaler: Scaler for sentiment features ('minmax', 'standard', or 'none')
            rngs: Random number generators
        """
        self.price_scaler_type = price_scaler
        self.volume_scaler_type = volume_scaler
        self.technical_scaler_type = technical_scaler
        self.sentiment_scaler_type = sentiment_scaler

        # Dictionaries to store scalers for each symbol
        self.price_scalers: Dict[str, ScalerModule] = {}
        self.volume_scalers: Dict[str, ScalerModule] = {}
        self.technical_scalers: Dict[str, ScalerModule] = {}
        self.sentiment_scalers: Dict[str, ScalerModule] = {}

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

            # 8. Additional features for RL model
            # Rate of change
            result["roc_5"] = ((result["close"] / result["close"].shift(5)) - 1) * 100
            result["roc_10"] = ((result["close"] / result["close"].shift(10)) - 1) * 100

            # Average true range (ATR) - volatility indicator
            tr1 = abs(result["high"] - result["low"])
            tr2 = abs(result["high"] - result["close"].shift(1))
            tr3 = abs(result["low"] - result["close"].shift(1))
            tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
            result["atr_14"] = tr.rolling(window=14).mean()

            # Money Flow Index (MFI) - volume-weighted RSI
            if "volume" in result.columns:
                typical_price = (result["high"] + result["low"] + result["close"]) / 3
                money_flow = typical_price * result["volume"]

                # Calculate positive and negative money flow
                pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
                neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

                # Calculate money flow ratio and index
                pos_flow_14 = pos_flow.rolling(window=14).sum()
                neg_flow_14 = neg_flow.rolling(window=14).sum()

                # Add small value to avoid division by zero
                money_ratio = pos_flow_14 / (neg_flow_14 + 1e-10)
                result["mfi_14"] = 100 - (100 / (1 + money_ratio))

        except Exception as e:
            logger.error(f"Error creating features: {e}")

        # Handle infinity values
        result = result.replace([np.inf, -np.inf], np.nan)

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

            # Create percent change target (useful for RL training)
            result[f"pct_change_{horizon}"] = (result[f"target_{horizon}"] / result[price_col]) - 1

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
                and not col.startswith("pct_change_")
            ]

            sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]

            # Initialize and fit price scaler
            if self.price_scaler_type.lower() != "none" and price_cols:
                # Create NNX scaler module if it doesn't exist
                if symbol not in self.price_scalers:
                    self.price_scalers[symbol] = ScalerModule(scaler_type=self.price_scaler_type)

                # Handle NaNs before fitting
                price_data = df[price_cols].dropna()
                # Handle outliers - clip extreme values
                for col in price_cols:
                    if col in price_data.columns:
                        # Use 3 standard deviations for clipping
                        mean = price_data[col].mean()
                        std = price_data[col].std()
                        if std > 0:  # Avoid division by zero
                            price_data[col] = price_data[col].clip(mean - 3 * std, mean + 3 * std)

                # Convert to numpy array for NNX scaler
                price_data_np = price_data.values
                self.price_scalers[symbol].fit(price_data_np)

            # Initialize and fit volume scaler
            if self.volume_scaler_type.lower() != "none" and volume_cols:
                # Create NNX scaler module if it doesn't exist
                if symbol not in self.volume_scalers:
                    self.volume_scalers[symbol] = ScalerModule(scaler_type=self.volume_scaler_type)

                # Handle NaNs and outliers before fitting
                volume_data = df[volume_cols].dropna()
                for col in volume_cols:
                    if col in volume_data.columns:
                        # Log transform for volume data (helps with extreme values)
                        volume_data[col] = np.log1p(volume_data[col].replace([np.inf, -np.inf, 0], np.nan).fillna(1))

                # Convert to numpy array for NNX scaler
                volume_data_np = volume_data.values
                self.volume_scalers[symbol].fit(volume_data_np)

            # Initialize and fit technical scaler
            if self.technical_scaler_type.lower() != "none" and technical_cols:
                # Create NNX scaler module if it doesn't exist
                if symbol not in self.technical_scalers:
                    self.technical_scalers[symbol] = ScalerModule(scaler_type=self.technical_scaler_type)

                # Handle NaNs and outliers before fitting
                technical_data = df[technical_cols].dropna()
                for col in technical_cols:
                    if col in technical_data.columns:
                        # Use 3 standard deviations for clipping
                        mean = technical_data[col].mean()
                        std = technical_data[col].std()
                        if std > 0:  # Avoid division by zero
                            technical_data[col] = technical_data[col].clip(mean - 3 * std, mean + 3 * std)

                # Convert to numpy array for NNX scaler
                technical_data_np = technical_data.values
                self.technical_scalers[symbol].fit(technical_data_np)

            # Initialize and fit sentiment scaler
            if self.sentiment_scaler_type.lower() != "none" and sentiment_cols:
                # Create NNX scaler module if it doesn't exist
                if symbol not in self.sentiment_scalers:
                    self.sentiment_scalers[symbol] = ScalerModule(scaler_type=self.sentiment_scaler_type)

                # Handle NaNs before fitting
                sentiment_data = df[sentiment_cols].fillna(0).dropna()

                # Convert to numpy array for NNX scaler
                sentiment_data_np = sentiment_data.values
                self.sentiment_scalers[symbol].fit(sentiment_data_np)

        except Exception as e:
            logger.error(f"Error fitting scalers for {symbol}: {e}")

    def transform_features(self, df: pd.DataFrame, symbol: str, is_train: bool = False) -> pd.DataFrame:
        """Transform features using fitted NNX scalers with improved handling of outliers and NaNs.

        Args:
            df: DataFrame with features
            symbol: Symbol for the data
            is_train: Whether this is training data

        Returns:
            DataFrame with transformed features
        """
        result = df.copy()

        try:
            # Handle outliers - clip extreme values
            numeric_cols = result.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if col in result.columns:
                    # Use 3 standard deviations for clipping to preserve trend information
                    # but remove extreme outliers that could affect scaling
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:  # Avoid division by zero
                        result[col] = result[col].clip(mean - 3 * std, mean + 3 * std)

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
                and not col.startswith("pct_change_")
            ]

            sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]

            # Transform price features
            if symbol in self.price_scalers and price_cols:
                price_scaler = self.price_scalers[symbol]
                # Use forward fill then backward fill to handle NaNs more robustly
                result[price_cols] = result[price_cols].ffill().bfill()
                # Replace any remaining NaNs or infs with 0
                result[price_cols] = result[price_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Check if data exists before transforming
                if not result[price_cols].empty:
                    # Convert to numpy array for NNX scaler
                    price_data_np = result[price_cols].values
                    transformed_data = price_scaler.transform(price_data_np)
                    result[price_cols] = transformed_data

            # Transform volume features
            if symbol in self.volume_scalers and volume_cols:
                volume_scaler = self.volume_scalers[symbol]
                # Apply same log transform as in fitting
                for col in volume_cols:
                    if col in result.columns:
                        result[col] = np.log1p(result[col].replace([np.inf, -np.inf, 0], np.nan).fillna(1))

                # Handle NaNs
                result[volume_cols] = result[volume_cols].ffill().bfill().fillna(0)

                # Check if data exists before transforming
                if not result[volume_cols].empty:
                    # Convert to numpy array for NNX scaler
                    volume_data_np = result[volume_cols].values
                    transformed_data = volume_scaler.transform(volume_data_np)
                    result[volume_cols] = transformed_data

            # Transform technical features
            if symbol in self.technical_scalers and technical_cols:
                technical_scaler = self.technical_scalers[symbol]
                # Handle NaNs
                result[technical_cols] = result[technical_cols].ffill().bfill().fillna(0)
                # Replace any remaining infs
                result[technical_cols] = result[technical_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Check if data exists before transforming
                if not result[technical_cols].empty:
                    # Convert to numpy array for NNX scaler
                    technical_data_np = result[technical_cols].values
                    transformed_data = technical_scaler.transform(technical_data_np)
                    result[technical_cols] = transformed_data

            # Transform sentiment features
            if symbol in self.sentiment_scalers and sentiment_cols:
                sentiment_scaler = self.sentiment_scalers[symbol]
                result[sentiment_cols] = result[sentiment_cols].fillna(0)  # Fill sentiment NaNs with 0

                # Check if data exists before transforming
                if not result[sentiment_cols].empty:
                    # Convert to numpy array for NNX scaler
                    sentiment_data_np = result[sentiment_cols].values
                    transformed_data = sentiment_scaler.transform(sentiment_data_np)
                    result[sentiment_cols] = transformed_data

            # Final NaN/inf check across all columns
            result = result.replace([np.inf, -np.inf], np.nan)
            # Forward fill, backward fill, then zero for any remaining NaNs
            result = result.ffill().bfill().fillna(0)

            # Add additional RL-specific normalization for better training stability
            if is_train:
                # Ensure important features for RL have consistent ranges
                important_cols = ["rsi_14", "macd", "bb_width", "volatility_20", "price_change", "volume_change"]
                for col in important_cols:
                    if col in result.columns and result[col].std() > 0:
                        # Additional normalization for key RL features if needed
                        if result[col].std() > 1.5 or result[col].std() < 0.1:
                            result[col] = (result[col] - result[col].mean()) / (result[col].std() + 1e-8)

        except Exception as e:
            logger.error(f"Error transforming features for {symbol}: {e}")

        return result
