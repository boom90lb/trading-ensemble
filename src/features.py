# src/features.py
"""Feature engineering for the time series ensemble model using NNX."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        # Train-only outlier-clip bounds keyed by (symbol, column). Populated in
        # fit_scalers and reused in transform_features so test-time clipping
        # never consumes test-set statistics.
        self.clip_bounds: Dict[str, Dict[str, tuple]] = {}
        # Column groups fitted per symbol. Transform uses these exact groups so
        # feature drift fails before it can silently pad/truncate scaler params.
        self.fitted_columns: Dict[str, Dict[str, List[str]]] = {}

    @staticmethod
    def _column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Return the feature groups used by fit/transform."""
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
        return {
            "price": price_cols,
            "volume": volume_cols,
            "technical": technical_cols,
            "sentiment": sentiment_cols,
        }

    @staticmethod
    def _scaler_state(scaler: ScalerModule) -> Dict[str, Any]:
        return {
            "scaler_type": scaler.scaler_type,
            "is_fitted": bool(scaler.is_fitted.value),
            "mean": np.asarray(scaler.mean.value, dtype=float),
            "std": np.asarray(scaler.std.value, dtype=float),
            "min_vals": np.asarray(scaler.min_vals.value, dtype=float),
            "max_vals": np.asarray(scaler.max_vals.value, dtype=float),
        }

    @staticmethod
    def _scaler_from_state(state: Dict[str, Any]) -> ScalerModule:
        scaler = ScalerModule(scaler_type=str(state["scaler_type"]))
        scaler.is_fitted.value = bool(state["is_fitted"])
        scaler.mean.value = np.asarray(state["mean"], dtype=float)
        scaler.std.value = np.asarray(state["std"], dtype=float)
        scaler.min_vals.value = np.asarray(state["min_vals"], dtype=float)
        scaler.max_vals.value = np.asarray(state["max_vals"], dtype=float)
        return scaler

    def to_plain_state(
        self,
        *,
        symbol: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Serialize fitted scaler state without depending on NNX internals."""
        symbols = [symbol] if symbol is not None else sorted(
            set(self.price_scalers)
            | set(self.volume_scalers)
            | set(self.technical_scalers)
            | set(self.sentiment_scalers)
            | set(self.clip_bounds)
            | set(self.fitted_columns)
        )

        def pick(store: Dict[str, ScalerModule]) -> Dict[str, Dict[str, Any]]:
            return {
                sym: self._scaler_state(store[sym])
                for sym in symbols
                if sym in store
            }

        return {
            "schema_version": 1,
            "price_scaler_type": self.price_scaler_type,
            "volume_scaler_type": self.volume_scaler_type,
            "technical_scaler_type": self.technical_scaler_type,
            "sentiment_scaler_type": self.sentiment_scaler_type,
            "price_scalers": pick(self.price_scalers),
            "volume_scalers": pick(self.volume_scalers),
            "technical_scalers": pick(self.technical_scalers),
            "sentiment_scalers": pick(self.sentiment_scalers),
            "clip_bounds": {
                sym: dict(self.clip_bounds.get(sym, {}))
                for sym in symbols
                if sym in self.clip_bounds
            },
            "fitted_columns": {
                sym: {
                    group: list(cols)
                    for group, cols in self.fitted_columns.get(sym, {}).items()
                }
                for sym in symbols
                if sym in self.fitted_columns
            },
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
        }

    @classmethod
    def from_plain_state(cls, state: Dict[str, Any]) -> "FeatureEngineer":
        """Rehydrate a FeatureEngineer from :meth:`to_plain_state` output."""
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError(
                f"Unsupported FeatureEngineer state schema: {state.get('schema_version')}"
            )
        fe = cls(
            price_scaler=str(state["price_scaler_type"]),
            volume_scaler=str(state["volume_scaler_type"]),
            technical_scaler=str(state["technical_scaler_type"]),
            sentiment_scaler=str(state["sentiment_scaler_type"]),
        )
        for sym, scaler_state in state.get("price_scalers", {}).items():
            fe.price_scalers[sym] = cls._scaler_from_state(scaler_state)
        for sym, scaler_state in state.get("volume_scalers", {}).items():
            fe.volume_scalers[sym] = cls._scaler_from_state(scaler_state)
        for sym, scaler_state in state.get("technical_scalers", {}).items():
            fe.technical_scalers[sym] = cls._scaler_from_state(scaler_state)
        for sym, scaler_state in state.get("sentiment_scalers", {}).items():
            fe.sentiment_scalers[sym] = cls._scaler_from_state(scaler_state)
        fe.clip_bounds = {
            sym: {col: tuple(bounds) for col, bounds in bounds_by_col.items()}
            for sym, bounds_by_col in state.get("clip_bounds", {}).items()
        }
        fe.fitted_columns = {
            sym: {group: list(cols) for group, cols in groups.items()}
            for sym, groups in state.get("fitted_columns", {}).items()
        }
        return fe

    def save_state(
        self,
        path: Path,
        *,
        symbol: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """Persist plain fold feature state to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                self.to_plain_state(symbol=symbol, feature_columns=feature_columns),
                f,
            )

    @classmethod
    def load_state(cls, path: Path) -> "FeatureEngineer":
        """Load plain fold feature state written by :meth:`save_state`."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls.from_plain_state(state)

    @staticmethod
    def _check_required_columns(
        df: pd.DataFrame,
        *,
        symbol: str,
        group_name: str,
        columns: List[str],
    ) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{symbol} {group_name} feature columns missing at transform: {missing}"
            )

    @staticmethod
    def _check_scaler_shape(
        scaler: ScalerModule,
        *,
        symbol: str,
        group_name: str,
        n_columns: int,
    ) -> None:
        if not bool(scaler.is_fitted.value):
            return
        if scaler.scaler_type == "minmax":
            fitted = int(np.asarray(scaler.min_vals.value).shape[0])
        elif scaler.scaler_type == "standard":
            fitted = int(np.asarray(scaler.mean.value).shape[0])
        else:
            return
        if fitted != n_columns:
            raise ValueError(
                f"{symbol} {group_name} scaler shape mismatch: "
                f"state has {fitted} columns, transform has {n_columns}"
            )

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
            groups = self._column_groups(df)
            price_cols = groups["price"]
            volume_cols = groups["volume"]
            technical_cols = groups["technical"]
            sentiment_cols = groups["sentiment"]
            self.fitted_columns[symbol] = {
                group: list(cols)
                for group, cols in groups.items()
            }

            # Train-only clip bounds for this symbol (overwritten on re-fit).
            symbol_bounds: Dict[str, tuple] = {}

            # Initialize and fit price scaler
            if self.price_scaler_type.lower() != "none" and price_cols:
                # Create NNX scaler module if it doesn't exist
                if symbol not in self.price_scalers:
                    self.price_scalers[symbol] = ScalerModule(scaler_type=self.price_scaler_type)

                # Handle NaNs before fitting
                price_data = df[price_cols].dropna()
                # Handle outliers - clip extreme values; record bounds for transform.
                for col in price_cols:
                    if col in price_data.columns:
                        mean = price_data[col].mean()
                        std = price_data[col].std()
                        if std > 0:
                            lo = float(mean - 3 * std)
                            hi = float(mean + 3 * std)
                            price_data[col] = price_data[col].clip(lo, hi)
                            symbol_bounds[col] = (lo, hi)

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
                        mean = technical_data[col].mean()
                        std = technical_data[col].std()
                        if std > 0:
                            lo = float(mean - 3 * std)
                            hi = float(mean + 3 * std)
                            technical_data[col] = technical_data[col].clip(lo, hi)
                            symbol_bounds[col] = (lo, hi)

                # Convert to numpy array for NNX scaler
                technical_data_np = technical_data.values
                # Log the shape for debugging
                logger.info(f"Technical data shape for {symbol}: {technical_data_np.shape}")
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

            # Commit train-only outlier bounds for downstream transform_features.
            self.clip_bounds[symbol] = symbol_bounds

        except Exception as e:
            logger.error(f"Error fitting scalers for {symbol}: {e}")
            raise

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
        groups = self.fitted_columns.get(symbol, self._column_groups(df))
        price_cols = list(groups.get("price", []))
        volume_cols = list(groups.get("volume", []))
        technical_cols = list(groups.get("technical", []))
        sentiment_cols = list(groups.get("sentiment", []))

        # Apply train-only outlier-clip bounds recorded in fit_scalers. Columns
        # without a stored bound pass through unclipped; missing fitted columns
        # are handled by the group-specific strict checks below.
        symbol_bounds = self.clip_bounds.get(symbol, {})
        for col, (lo, hi) in symbol_bounds.items():
            if col in result.columns:
                result[col] = result[col].clip(lo, hi)

        if symbol in self.price_scalers and price_cols:
            self._check_required_columns(
                result, symbol=symbol, group_name="price", columns=price_cols,
            )
            price_scaler = self.price_scalers[symbol]
            result[price_cols] = (
                result[price_cols]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0)
            )
            price_data_np = result[price_cols].values
            self._check_scaler_shape(
                price_scaler, symbol=symbol, group_name="price",
                n_columns=price_data_np.shape[1],
            )
            result[price_cols] = price_scaler.transform(price_data_np)

        if symbol in self.volume_scalers and volume_cols:
            self._check_required_columns(
                result, symbol=symbol, group_name="volume", columns=volume_cols,
            )
            volume_scaler = self.volume_scalers[symbol]
            for col in volume_cols:
                result[col] = np.log1p(
                    result[col].replace([np.inf, -np.inf, 0], np.nan).fillna(1)
                )
            result[volume_cols] = result[volume_cols].ffill().fillna(0)
            volume_data_np = result[volume_cols].values
            self._check_scaler_shape(
                volume_scaler, symbol=symbol, group_name="volume",
                n_columns=volume_data_np.shape[1],
            )
            result[volume_cols] = volume_scaler.transform(volume_data_np)

        if symbol in self.technical_scalers and technical_cols:
            self._check_required_columns(
                result, symbol=symbol, group_name="technical",
                columns=technical_cols,
            )
            technical_scaler = self.technical_scalers[symbol]
            result[technical_cols] = (
                result[technical_cols]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0)
            )
            technical_data_np = result[technical_cols].values
            self._check_scaler_shape(
                technical_scaler, symbol=symbol, group_name="technical",
                n_columns=technical_data_np.shape[1],
            )
            result[technical_cols] = technical_scaler.transform(technical_data_np)

        if symbol in self.sentiment_scalers and sentiment_cols:
            self._check_required_columns(
                result, symbol=symbol, group_name="sentiment",
                columns=sentiment_cols,
            )
            sentiment_scaler = self.sentiment_scalers[symbol]
            result[sentiment_cols] = (
                result[sentiment_cols]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0)
            )
            sentiment_data_np = result[sentiment_cols].values
            self._check_scaler_shape(
                sentiment_scaler, symbol=symbol, group_name="sentiment",
                n_columns=sentiment_data_np.shape[1],
            )
            result[sentiment_cols] = sentiment_scaler.transform(sentiment_data_np)

        result = result.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        if is_train:
            # Preserve the existing train-only RL stabilization behavior.
            important_cols = ["rsi_14", "macd", "bb_width", "volatility_20", "price_change", "volume_change"]
            for col in important_cols:
                if col in result.columns and result[col].std() > 0:
                    if result[col].std() > 1.5 or result[col].std() < 0.1:
                        result[col] = (result[col] - result[col].mean()) / (result[col].std() + 1e-8)

        return result
