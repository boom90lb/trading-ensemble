# src/baselines/__init__.py
"""Baseline strategies for fair comparison against the ensemble.

Each baseline implements `BaseModel` and emits positions in [-1, 1]. Baselines
are stateless rule-based mappings from a close series to a position series;
they precompute the full series in `prepare(close)` and `predict(X)` is a
date lookup. Each instance binds to one symbol's close series, so multi-symbol
backtests instantiate one baseline per (model_class, symbol).
"""

from src.baselines.buy_and_hold import BuyAndHold
from src.baselines.ma_crossover import MACrossover
from src.baselines.tsmom import TSMOM

__all__ = ["BuyAndHold", "MACrossover", "TSMOM"]
