# src/data_loader.py
"""Data loading module for the time series ensemble model."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from src.config import DATA_DIR, TWELVEDATA_API_KEY, TrainingConfig

logger = logging.getLogger(__name__)

# Bar DatetimeIndex is localized to America/New_York. The trading-day calendar is
# ET; sentiment_analysis joins UTC news against (idx + 16h) as the bar close time.
BAR_TZ = "America/New_York"
BAR_CLOSE_HOUR = 16


def _ensure_bar_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Localize a price DataFrame's DatetimeIndex to BAR_TZ if it is naive."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(BAR_TZ, nonexistent="shift_forward", ambiguous="NaT")
        df = df[df.index.notna()]
    elif str(df.index.tz) != BAR_TZ:
        df.index = df.index.tz_convert(BAR_TZ)
    return df


def _tz_aware_filter(ts: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse a YYYY-MM-DD string into a tz-aware ET Timestamp for index filtering."""
    if ts is None:
        return None
    return pd.Timestamp(ts).tz_localize(BAR_TZ)


# Cache filenames encode the *requested* date range so a file's coverage is
# visible from its name (Phase 4 §4.4). A None start (no lower bound) encodes
# as this sentinel.
CACHE_START_SENTINEL = "min"


def _parse_cache_range(filename: str, symbol: str, interval: str):
    """Parse ``{symbol}_{interval}_{start}_{end}.parquet`` -> (start, end).

    Returns the requested-range tokens the file was fetched for (``start`` is
    ``None`` when the ``min`` sentinel was used), or ``None`` if the name does
    not match the range-encoded scheme (e.g. a legacy ``{symbol}_{interval}``
    file, which is intentionally ignored so its unknown coverage can't be
    silently trusted).
    """
    prefix = f"{symbol}_{interval}_"
    suffix = ".parquet"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    middle = filename[len(prefix) : -len(suffix)]
    parts = middle.split("_")
    if len(parts) != 2:
        return None
    start_token, end_token = parts
    start = None if start_token == CACHE_START_SENTINEL else start_token
    return start, end_token


def _range_covers(
    cov_start: Optional[str], cov_end: str, req_start: Optional[str], req_end: str
) -> bool:
    """Does a cached file's *requested* range [cov_start, cov_end] contain the
    new request [req_start, req_end]?

    Coverage is decided on the requested ranges (from the filename), NOT on the
    data's actual min/max: a short-history symbol (late IPO, recent listing)
    would otherwise force an endless refetch because the API can't return bars
    before the listing date, so ``index.min()`` is permanently > the requested
    start. A ``None`` start means "from the beginning" and is only covered by
    another ``None`` (sentinel ``min``) start. ISO ``YYYY-MM-DD`` strings sort
    lexicographically as dates, so plain ``<`` comparison is correct.
    """
    if cov_start is not None:
        if req_start is None or req_start < cov_start:
            return False
    if cov_end < req_end:
        return False
    return True


def _parse_dividends_payload(data) -> pd.Series:
    """Twelvedata ``/dividends`` JSON -> tz-aware ET Series (ex_date -> amount).

    Returns an empty Series for any error / no-dividend / unexpected-schema
    response so the caller can treat "no dividend data" as a backtest no-op.
    Amounts are split-adjusted to match ``adjust=splits`` prices. Same-ex-date
    rows (e.g. regular + special dividend) are summed.
    """
    empty = pd.Series(dtype="float64", name="amount")
    if not isinstance(data, dict) or data.get("status") == "error":
        return empty
    records = data.get("dividends") or []
    if not records:
        return empty
    df = pd.DataFrame(records)
    if "ex_date" not in df.columns or "amount" not in df.columns:
        logger.error(f"Unexpected /dividends schema: {list(df.columns)}")
        return empty
    amount = pd.to_numeric(df["amount"], errors="coerce")
    idx = pd.to_datetime(df["ex_date"], errors="coerce")
    s = pd.Series(amount.to_numpy(), index=idx, name="amount").dropna()
    s = s[s.index.notna()]
    if s.empty:
        return empty
    s = s.groupby(level=0).sum().sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(BAR_TZ)
    return s


# Twelvedata's time_series endpoint expects interval strings like "1day"/
# "1week"/"1month"; the project (config defaults, cache filenames, CLI) uses
# the shorter "1d"/"1wk"/"1mo" convention. Normalize ONLY at the API boundary
# so cache keys and config stay in the project convention. Intraday strings
# ("1h", "2h", ...) already match the vendor and pass through unchanged.
_VENDOR_INTERVALS = {"1d": "1day", "1wk": "1week", "1mo": "1month"}


def _to_vendor_interval(interval: str) -> str:
    """Map the project's interval shorthand to Twelvedata's expected string."""
    return _VENDOR_INTERVALS.get(interval, interval)


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

    def _find_covering_cache(
        self, symbol: str, interval: str, req_start: Optional[str], req_end: str
    ) -> Optional[Path]:
        """Return a cached parquet whose requested range covers [start, end].

        Scans range-encoded cache files for this (symbol, interval) and returns
        the first whose fetched range contains the new request. Legacy
        un-ranged files are skipped (their coverage is unknown), forcing a
        refetch rather than silently returning a too-narrow slice.
        """
        pattern = f"{symbol}_{interval}_*.parquet"
        for path in sorted(self.cache_dir.glob(pattern)):
            parsed = _parse_cache_range(path.name, symbol, interval)
            if parsed is None:
                continue
            cov_start, cov_end = parsed
            if _range_covers(cov_start, cov_end, req_start, req_end):
                return path
        return None

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

        Corporate actions (M6, Phase 4 §4.1):
            Prices are fetched with ``adjust=splits`` — split-adjusted but NOT
            dividend-adjusted, so the series is a faithful tradeable price.
            Dividends are handled as explicit cash in the backtest (see
            ``fetch_dividends`` + ``TradingStrategy.apply_dividends``): a long
            held over an ex-date shows a markdown loss in mark-to-market that
            the dividend credit offsets, making the position total-return
            correct without back-adjusting (and back-adjustment's price-level
            look-ahead). Trade-off: ex-div gaps remain in the return/vol
            *features*, which is the accepted cost of this choice.
        """
        # Set end date to today if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Look for a cached file whose *requested* range covers [start, end].
        # Coverage is checked by filename range (Phase 4 §4.4) so a narrow cache
        # can no longer silently satisfy a wider request by returning a slice.
        cache_hit = self._find_covering_cache(symbol, interval, start_date, end_date)
        if cache_hit is not None:
            try:
                df = pd.read_parquet(cache_hit)
                df.index = pd.to_datetime(df.index)
                df = _ensure_bar_tz(df)

                # Filter by date if needed
                start_ts = _tz_aware_filter(start_date)
                end_ts = _tz_aware_filter(end_date)
                if start_ts is not None:
                    df = df[df.index >= start_ts]
                if end_ts is not None:
                    df = df[df.index <= end_ts]

                # Non-empty means the covering file actually held bars in range.
                # Empty (corrupt/holiday-only) falls through to a refetch.
                if not df.empty:
                    return df
                logger.info(
                    f"Cache {cache_hit.name} empty after date filter; refetching {symbol}"
                )

            except Exception as e:
                logger.error(f"Error reading cache file {cache_hit}: {e}")

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
                "interval": _to_vendor_interval(interval),
                "apikey": self.api_key,
                "format": "json",
                "outputsize": output_size,
                # Phase 4 §4.1: split-adjusted (deterministic, not relying on
                # the vendor default); dividends credited as cash downstream.
                "adjust": "splits",
            }

            # Add date parameters if provided
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            # Make API request
            response = requests.get("https://api.twelvedata.com/time_series", params=params)  # type: ignore
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
            df = _ensure_bar_tz(df)

            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]  # type: ignore

            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Cache under a range-encoded filename so coverage is visible from
            # the name and a future wider request triggers a refetch instead of
            # trusting a narrow slice (Phase 4 §4.4).
            start_token = start_date if start_date else CACHE_START_SENTINEL
            cache_file = self.cache_dir / f"{symbol}_{interval}_{start_token}_{end_date}.parquet"
            try:
                df.to_parquet(cache_file)
                logger.info(f"Cached data for {symbol} to {cache_file}")
            except Exception as e:
                logger.error(f"Error caching data to {cache_file}: {e}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_dividends(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch per-share cash dividends (ex-date -> amount) from Twelvedata.

        Returns a tz-aware (ET) Series indexed by ex-date, split-adjusted to
        match the ``adjust=splits`` prices from ``fetch_historical_data`` so
        the cash credited per (split-adjusted) share is consistent (Phase 4
        §4.1). Empty Series when the symbol pays no dividends in range or the
        API is unavailable — the backtest treats that as a no-op. Cached with
        the same range-encoded scheme as prices (``{symbol}_dividends_*``); an
        empty result is a valid cached answer (not refetched).
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_hit = self._find_covering_cache(symbol, "dividends", start_date, end_date)
        if cache_hit is not None:
            try:
                cached = pd.read_parquet(cache_hit)
                series = self._dividends_frame_to_series(cached)
                return self._filter_div_series(series, start_date, end_date)
            except Exception as e:
                logger.error(f"Error reading dividend cache {cache_hit}: {e}")

        series = self._request_dividends(symbol, start_date, end_date)

        start_token = start_date if start_date else CACHE_START_SENTINEL
        cache_file = self.cache_dir / f"{symbol}_dividends_{start_token}_{end_date}.parquet"
        try:
            series.to_frame("amount").to_parquet(cache_file)
        except Exception as e:
            logger.error(f"Error caching dividends to {cache_file}: {e}")

        return self._filter_div_series(series, start_date, end_date)

    def _request_dividends(
        self, symbol: str, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.Series:
        """Hit ``/dividends`` and parse; empty Series on any failure."""
        empty = pd.Series(dtype="float64", name="amount")
        if not self.api_key:
            logger.warning(f"No API key; cannot fetch dividends for {symbol}")
            return empty
        params = {"symbol": symbol, "apikey": self.api_key, "format": "json"}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        try:
            response = requests.get("https://api.twelvedata.com/dividends", params=params)
            response.raise_for_status()
            return _parse_dividends_payload(response.json())
        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return empty

    @staticmethod
    def _dividends_frame_to_series(df: pd.DataFrame) -> pd.Series:
        """Round-trip a cached one-column ``amount`` frame back to a Series."""
        if df.empty or "amount" not in df.columns:
            return pd.Series(dtype="float64", name="amount")
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(BAR_TZ)
        return pd.Series(df["amount"].astype(float).to_numpy(), index=idx, name="amount")

    @staticmethod
    def _filter_div_series(
        series: pd.Series, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.Series:
        if series.empty:
            return series
        start_ts = _tz_aware_filter(start_date)
        end_ts = _tz_aware_filter(end_date)
        if start_ts is not None:
            series = series[series.index >= start_ts]
        if end_ts is not None:
            series = series[series.index <= end_ts]
        return series

    def fetch_training_data(
        self, config: TrainingConfig
    ) -> Dict[str, pd.DataFrame]:
        """Fetch full per-symbol OHLCV frames for WFO training.

        Phase 2.2: train/test splitting is now handled by the WFO outer
        loop via ``PurgedWalkForward``, so this returns one frame per
        symbol with no preemptive split. Symbols that come back empty are
        omitted from the result dict.
        """
        all_data: Dict[str, pd.DataFrame] = {}

        for symbol in config.symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.fetch_historical_data(
                symbol=symbol,
                interval=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
            )
            if df.empty:
                logger.warning(f"No data fetched for {symbol}, skipping")
                continue
            all_data[symbol] = df

        return all_data
