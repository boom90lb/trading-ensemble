# src/data_loader.py
"""Data loading module for the time series ensemble model."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from src.config import DATA_DIR, TWELVEDATA_API_KEY, TrainingConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for fetching and preparing time series data."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """Initialize the data loader.

        Args:
            api_key: Twelvedata API key (default: use from config)
            cache_dir: Directory for caching data (default: use from config)
        """
        self.api_key = api_key or TWELVEDATA_API_KEY
        if not self.api_key:
            logger.warning("No Twelvedata API key provided, data fetching will be limited")

        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def fetch_historical_data(
        self, symbol: str, interval: str = "1d", start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical price data from Twelvedata or cache.

        Args:
            symbol: Ticker symbol
            interval: Time interval (e.g., 1d, 1h)
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)

        Returns:
            DataFrame with historical price data
        """
        # Prepare cache file path
        cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"

        # Set end date to today if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Check if we have cached data
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)

                # Filter by date if needed
                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]

                # If we have sufficient data, return it
                if not df.empty:
                    return df

            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")

        # Fetch data from API if no cached data or dates are outside cached range
        try:
            if not self.api_key:
                raise ValueError("API key is required for fetching data")

            # Set output size based on interval to get enough data
            if interval in ["1d", "1wk", "1mo"]:
                output_size = 5000  # Maximum available
            else:
                output_size = 5000  # Use a reasonable default for intraday

            # Prepare API parameters
            params = {
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "format": "json",
                "outputsize": output_size,
            }

            # Add date parameters if provided
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            # Make API request
            response = requests.get("https://api.twelvedata.com/time_series", params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "Unknown API error")
                logger.error(f"API error for {symbol}: {error_msg}")
                return pd.DataFrame()

            # Process response
            if "values" not in data:
                logger.error(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Create DataFrame from values
            values = data["values"]
            df = pd.DataFrame(values)

            # Convert types and set index
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df = df.sort_index()

            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Cache the data
            try:
                df.to_parquet(cache_file)
                logger.info(f"Cached data for {symbol} to {cache_file}")
            except Exception as e:
                logger.error(f"Error caching data to {cache_file}: {e}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_and_prepare_training_data(
        self, config: TrainingConfig
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """Fetch and prepare data for model training.

        Args:
            config: Training configuration

        Returns:
            Tuple of (all_data, train_test_splits)
            - all_data: Dictionary of {symbol: full_dataframe}
            - train_test_splits: Dictionary of {symbol: (train_df, test_df)}
        """
        all_data = {}
        train_test_splits = {}

        for symbol in config.symbols:
            logger.info(f"Fetching data for {symbol}")

            # Fetch historical data
            df = self.fetch_historical_data(
                symbol=symbol, interval=config.timeframe, start_date=config.start_date, end_date=config.end_date
            )

            if df.empty:
                logger.warning(f"No data fetched for {symbol}, skipping")
                continue

            # Store the full dataset
            all_data[symbol] = df

            # Create train/test split
            if config.train_test_split > 0 and config.train_test_split < 1:
                # Calculate split index
                split_idx = int(len(df) * config.train_test_split)

                # Split the data
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()

                # Store the split
                train_test_splits[symbol] = (train_df, test_df)

                logger.info(f"Created train/test split for {symbol}: train={len(train_df)}, test={len(test_df)}")
            else:
                # Use all data for both training and testing if split is invalid
                logger.warning(f"Invalid train_test_split={config.train_test_split}, using all data")
                train_test_splits[symbol] = (df.copy(), df.copy())

        return all_data, train_test_splits
