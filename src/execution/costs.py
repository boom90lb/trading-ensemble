"""Pure transaction-cost functions.

All bps inputs are basis points (1 bp = 0.01%). All dollar outputs are positive
magnitudes — sign conventions live in the caller (ExecutionModel).
"""


def slippage_bps(
    trade_notional: float,
    portfolio_value: float,
    half_spread_bps: float,
    slippage_coeff: float,
) -> float:
    """Total one-way execution-cost in bps for a trade of given $notional.

    Composed of two parts:
      * `half_spread_bps`: constant — half the quoted bid-ask spread.
      * Linear price-impact: `slippage_coeff * |trade_notional| / portfolio_value`.
        Coeff=10 ⇒ a 100%-of-portfolio trade adds +10 bps on top of spread.

    A degenerate `portfolio_value <= 0` (margin call, fully drawn down) collapses
    impact to zero — the strategy is dead by that point and the fill itself
    is academic.
    """
    impact = 0.0
    if portfolio_value > 0:
        impact = slippage_coeff * abs(trade_notional) / portfolio_value
    return half_spread_bps + impact


def commission_dollars(trade_notional: float, commission_bps: float) -> float:
    """One-way commission in $ as bps of |notional|. Always non-negative."""
    return abs(trade_notional) * commission_bps / 10_000.0


def daily_borrow_dollars(
    short_notional: float,
    borrow_rate_bps_annual: float,
    trading_days_per_year: int = 252,
) -> float:
    """Per-bar borrow charge in $ on a held short position.

    `short_notional` is the absolute $ value of the open short. Returns a
    positive number to be subtracted from cash. Zero if no short held.
    """
    if short_notional <= 0:
        return 0.0
    annual_rate = borrow_rate_bps_annual / 10_000.0
    return short_notional * annual_rate / trading_days_per_year
