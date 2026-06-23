"""Shared data-loading helpers for statistical-arbitrage CLIs."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data_loader import DataLoader
from src.logging_utils import get_symbol_logger


def fetch_price_matrices(
    symbols: List[str], start_date: str, end_date: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader()
    closes: Dict[str, pd.Series] = {}
    opens: Dict[str, pd.Series] = {}
    for symbol in symbols:
        log = get_symbol_logger(__name__, symbol)
        df = loader.fetch_historical_data(symbol, "1d", start_date, end_date)
        if df.empty:
            log.warning("no bars returned; skipping")
            continue
        required = {"open", "close"}
        missing = required - set(df.columns)
        if missing:
            log.warning("missing required columns %s; skipping", sorted(missing))
            continue
        closes[symbol] = df["close"].astype(float)
        opens[symbol] = df["open"].astype(float)
    if len(closes) < 2:
        raise RuntimeError("Need at least two symbols with open/close bars for stat-arb.")
    close_matrix = pd.DataFrame(closes).sort_index().dropna(how="all")
    open_matrix = pd.DataFrame(opens).sort_index().reindex(close_matrix.index).dropna(how="all")
    return close_matrix, open_matrix
