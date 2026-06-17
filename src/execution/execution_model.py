"""ExecutionModel: order queue + fill timing + per-fill cost accounting.

Order timing convention
-----------------------
Signals are generated using data through the close of bar t. Orders
submitted "on bar t" therefore reach the market at end-of-bar t. The two
supported order types both fill on the next bar, t+1:

  * MOO (market-on-open): fill at open(t+1).
  * MOC (market-on-close): fill at close(t+1).

This eliminates same-bar fill leakage: the bar that produces the signal
is never the bar that fills the order. Unfilled orders at fold boundaries
(submitted in fold k with no t+1 bar inside fold k) are dropped — they
must not silently spill into fold k+1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping

from src.config import ExecutionConfig
from src.execution.costs import commission_dollars, slippage_bps

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MOO = "MOO"
    MOC = "MOC"


@dataclass
class Order:
    """A directed order awaiting execution.

    `shares` is always positive; `side` carries the direction:
      * +1 for buy (open/cover)
      * -1 for sell (close/short)
    """

    symbol: str
    side: int
    shares: float
    order_type: OrderType
    submit_bar: int

    def __post_init__(self) -> None:
        assert self.side in (-1, 1), f"side must be ±1; got {self.side}"
        assert self.shares > 0, f"shares must be > 0; got {self.shares}"


@dataclass
class Fill:
    """The realized execution of an Order, with cost breakdown."""

    order: Order
    fill_bar: int
    fill_price: float
    # Signed: + for buys (cash out), - for sells (cash in), before costs.
    notional: float
    commission: float
    slippage_cost: float


@dataclass
class ExecutionModel:
    """In-memory order queue with bps-cost fills.

    The model is symbol-aware but does not maintain portfolio state — the
    caller (Trader) tracks cash and positions and applies returned Fills.
    """

    config: ExecutionConfig
    _pending: List[Order] = field(default_factory=list)

    def submit(self, order: Order) -> None:
        self._pending.append(order)

    def pending_count(self) -> int:
        return len(self._pending)

    def drop_pending(self) -> int:
        """Discard all queued orders (use at fold boundaries). Returns count dropped."""
        n = len(self._pending)
        if n:
            logger.info("ExecutionModel: dropping %d unfilled order(s) at boundary", n)
        self._pending = []
        return n

    def step(
        self,
        bar_idx: int,
        bar_ohlc: Mapping[str, Mapping[str, float]],
        portfolio_value: float,
    ) -> List[Fill]:
        """Process pending orders against bar `bar_idx`.

        Args:
            bar_idx: integer index of the bar being processed.
            bar_ohlc: {symbol: {"open": float, "close": float, ...}} for this bar.
            portfolio_value: current portfolio value, used to scale linear impact.

        Returns:
            Fills for every order submitted before ``bar_idx`` with OHLC on
            this bar. Orders missing OHLC for their symbol remain queued.
        """
        fills: List[Fill] = []
        still_pending: List[Order] = []

        for order in self._pending:
            if order.submit_bar >= bar_idx:
                still_pending.append(order)
                continue

            sym_bar = bar_ohlc.get(order.symbol)
            if sym_bar is None:
                # No price for this symbol on this bar — keep waiting. (E.g.,
                # half-day, holiday, or a per-symbol data gap.)
                still_pending.append(order)
                continue

            ref_price_key = "open" if order.order_type == OrderType.MOO else "close"
            ref_price = float(sym_bar[ref_price_key])

            raw_notional = order.side * order.shares * ref_price
            slip_bps = slippage_bps(
                trade_notional=raw_notional,
                portfolio_value=portfolio_value,
                half_spread_bps=self.config.spread_bps,
                slippage_coeff=self.config.slippage_coeff,
            )
            # Buys fill above ref, sells fill below — adverse in both cases.
            fill_price = ref_price * (1.0 + order.side * slip_bps / 10_000.0)
            fill_notional = order.side * order.shares * fill_price
            slippage_cost = abs(raw_notional) * slip_bps / 10_000.0
            commission = commission_dollars(raw_notional, self.config.commission_bps)

            fills.append(
                Fill(
                    order=order,
                    fill_bar=bar_idx,
                    fill_price=fill_price,
                    notional=fill_notional,
                    commission=commission,
                    slippage_cost=slippage_cost,
                )
            )

        self._pending = still_pending
        return fills
