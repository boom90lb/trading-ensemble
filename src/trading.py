# src/trading.py
"""Trading strategy and backtesting module."""

import logging
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from src.config import RESULTS_DIR, TradingConfig
from src.execution import ExecutionModel, Fill, Order, OrderType
from src.execution.costs import daily_borrow_dollars
from src.execution.target_weights import scale_to_max_gross
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.models.mapping import realized_vol
from src.sentiment_analysis import SentimentAnalyzer
from src.validation.metrics import calmar_ratio, probabilistic_sharpe_ratio

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
        execution_model: Optional[ExecutionModel] = None,
    ):
        """Initialize the trading strategy.

        Args:
            model: Trained forecasting model
            config: Trading configuration
            sentiment_analyzer: Sentiment analyzer for news data
            use_sentiment: Whether to use sentiment data in trading decisions
            execution_model: ExecutionModel handling order queue + fills + costs.
                Defaults to one constructed from `config.execution`. Inject a
                custom model in tests or to share a queue across symbols.
        """
        self.model = model
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.use_sentiment = use_sentiment and sentiment_analyzer is not None
        self.execution_model = execution_model or ExecutionModel(config.execution)
        self.default_order_type = OrderType(config.execution.default_order_type)
        self.positions: Dict[str, PositionInfo] = {}
        self.cash = config.initial_capital
        self.portfolio_value = config.initial_capital
        self.history: List[Dict[str, Union[pd.Timestamp, float]]] = []
        self.transaction_log: List[Dict[str, Any]] = []
        # Maps bar_idx -> Timestamp; populated during backtest for transaction logging.
        self._bar_dates: Dict[int, pd.Timestamp] = {}

    def calculate_signal(
        self,
        predictions: np.ndarray,
        current_price: float,
        symbol: str,
        date: pd.Timestamp,
        sentiment_data: Optional[pd.DataFrame] = None,
        band_width: Optional[float] = None,
        features: Optional[pd.DataFrame] = None,
        policy_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[Position, float]:
        """Translate an ensemble position ∈ [-1, 1] into a discrete trade.

        After Phase 1.2 the ensemble emits unified positions, so this function
        no longer re-derives expected return from price. Sign → LONG/SHORT/FLAT;
        magnitude → confidence. `current_price` is kept in the signature for
        downstream callers but is no longer used for signal direction.

        ``features`` is the same per-bar window fed to ``model.predict``; it is
        forwarded to ``get_model_contributions`` for the inter-model-agreement
        confidence boost. Passing it is required — an empty frame makes the
        ensemble's vol-sizing raise "requires 'close'" and the boost no-ops.

        Phase 2.5: when band_width is provided (from conformal predict_band),
        scales confidence down by the width of the position-space band. Wider
        bands (high uncertainty) reduce confidence; tight bands increase it.
        """
        del current_price  # signal is derived from the position directly now

        if len(predictions) == 0:
            logger.warning("No predictions available for signal calculation")
            return Position.FLAT, 0.5

        position_target = float(np.clip(predictions[0], -1.0, 1.0))

        # Confidence: start from |position|, boost on inter-model agreement.
        confidence = abs(position_target)
        if isinstance(self.model, EnsembleModel) and features is not None and not features.empty:
            try:
                contributions = self.model.get_model_contributions(
                    features,
                    policy_X=policy_features,
                )
                if not contributions.empty:
                    # Use the LAST row — the current bar's decision — to match
                    # how predict()'s latest_prediction is taken.
                    member_positions = [
                        contributions[f"{name}_position"].iloc[-1]
                        for name in self.model.models.keys()
                        if f"{name}_position" in contributions.columns
                    ]
                    signs = [np.sign(p) for p in member_positions if not np.isnan(p) and p != 0]
                    if signs:
                        agreement = abs(float(np.mean(signs)))
                        confidence = 0.5 * confidence + 0.5 * agreement
            except Exception as e:
                logger.error(f"Error calculating model contributions: {e}")

        if self.use_sentiment and sentiment_data is not None:
            try:
                sentiment_score = (
                    float(sentiment_data[f"{symbol}_sentiment_score"].iloc[0])
                    if f"{symbol}_sentiment_score" in sentiment_data.columns
                    else 0.0
                )
                sentiment_momentum = (
                    float(sentiment_data[f"{symbol}_sentiment_momentum"].iloc[0])
                    if f"{symbol}_sentiment_momentum" in sentiment_data.columns
                    else 0.0
                )
                sentiment_factor = 0.3
                sentiment_pos = np.clip((sentiment_score + sentiment_momentum) / 3.0, -1.0, 1.0)
                position_target = float(
                    np.clip(
                        (1.0 - sentiment_factor) * position_target + sentiment_factor * sentiment_pos,
                        -1.0,
                        1.0,
                    )
                )
                sentiment_strength = abs(sentiment_score) + abs(sentiment_momentum)
                confidence = min(1.0, confidence + min(0.2, sentiment_strength / 10.0))
            except Exception as e:
                logger.error(f"Error incorporating sentiment data: {e}")

        # Phase 2.5: scale confidence by conformal band width. Band width is
        # in position space [0, 2*position_cap]; normalize and apply as a
        # tightness factor. Tight bands (high confidence) boost; wide bands
        # (high uncertainty) dampen confidence.
        if band_width is not None:
            max_band = 2.0 * self.model.position_cap if isinstance(
                self.model, EnsembleModel
            ) else 2.0
            band_factor = max(0.0, 1.0 - band_width / max_band)
            confidence *= band_factor

        # Minimum magnitude to avoid overtrading on noise. Default 0.1 ≈ 10% of
        # full bet — matches the spirit of the old 0.2% return threshold scaled
        # by a typical 2% daily vol. Configurable (Phase 2.7) so a sweep can
        # vary it.
        threshold = self.config.signal_threshold

        if position_target > threshold:
            return Position.LONG, min(1.0, max(confidence, position_target))
        elif position_target < -threshold:
            return Position.SHORT, min(1.0, max(confidence, abs(position_target)))
        else:
            return Position.FLAT, confidence

    def calculate_target_weight(
        self,
        predictions: np.ndarray,
        current_price: float,
        symbol: str,
        date: pd.Timestamp,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """Translate a model position into a continuous portfolio target weight.

        Target-weight mode intentionally bypasses ``signal_threshold``. The
        model's signed position is scaled by ``position_size``; no trade/no-trade
        decision is made here because the portfolio accounting layer suppresses
        small rebalances against the currently filled weight.
        """
        del current_price, date

        if len(predictions) == 0:
            logger.warning("No predictions available for target-weight calculation")
            return 0.0

        position_target = float(np.clip(predictions[0], -1.0, 1.0))
        if self.use_sentiment and sentiment_data is not None:
            try:
                sentiment_score = (
                    float(sentiment_data[f"{symbol}_sentiment_score"].iloc[0])
                    if f"{symbol}_sentiment_score" in sentiment_data.columns
                    else 0.0
                )
                sentiment_momentum = (
                    float(sentiment_data[f"{symbol}_sentiment_momentum"].iloc[0])
                    if f"{symbol}_sentiment_momentum" in sentiment_data.columns
                    else 0.0
                )
                sentiment_factor = 0.3
                sentiment_pos = np.clip((sentiment_score + sentiment_momentum) / 3.0, -1.0, 1.0)
                position_target = float(
                    np.clip(
                        (1.0 - sentiment_factor) * position_target + sentiment_factor * sentiment_pos,
                        -1.0,
                        1.0,
                    )
                )
            except Exception as e:
                logger.error(f"Error incorporating sentiment data: {e}")

        return float(self.config.position_size * position_target)

    def submit_signal_order(
        self,
        symbol: str,
        position: Position,
        confidence: float,
        signal_price: float,
        bar_idx: int,
        order_type: Optional[OrderType] = None,
    ) -> None:
        """Translate a target Position+confidence into queued Order(s).

        A flip from LONG↔SHORT becomes two orders: a closer for the old
        position, then an opener at the new side. Orders are sized from
        `signal_price` (the close at signal time); the fill price differs.

        Args:
            symbol: Symbol to trade.
            position: Target position after the orders fill.
            confidence: Confidence in the signal in [0, 1].
            signal_price: Price at signal time (typically bar's close).
                Used to estimate shares; actual fill price set by ExecutionModel.
            bar_idx: Integer index of the signal bar; orders fill at bar_idx+1.
            order_type: Override default order type for this signal.
        """
        order_type = order_type or self.default_order_type
        current = self.positions.get(
            symbol, {"position": Position.FLAT, "entry_price": 0.0, "size": 0.0}
        )

        if current["position"] == position:
            return

        # Close any existing position (full close — sizing is recomputed on open).
        if current["position"] != Position.FLAT:
            close_side = -1 if current["position"] == Position.LONG else 1
            self.execution_model.submit(
                Order(
                    symbol=symbol,
                    side=close_side,
                    shares=float(current["size"]),
                    order_type=order_type,
                    submit_bar=bar_idx,
                )
            )

        # Open new position.
        if position != Position.FLAT:
            position_size = self.config.position_size * confidence
            max_allocation = self.portfolio_value * position_size
            shares = max_allocation / signal_price
            if shares <= 0:
                return
            open_side = 1 if position == Position.LONG else -1
            self.execution_model.submit(
                Order(
                    symbol=symbol,
                    side=open_side,
                    shares=shares,
                    order_type=order_type,
                    submit_bar=bar_idx,
                )
            )

    def apply_fills(self, fills: List[Fill]) -> None:
        """Apply ExecutionModel fills to cash + positions, emit transaction log.

        Disambiguates intent (open vs close, long vs short) by inspecting the
        current position state *before* each fill — fills are processed in
        submission order, so a LONG→SHORT flip resolves as CLOSE_LONG then
        OPEN_SHORT even though both fills have side=-1.
        """
        for fill in fills:
            order = fill.order
            symbol = order.symbol
            shares = order.shares
            fill_price = fill.fill_price
            commission = fill.commission
            slippage_cost = fill.slippage_cost
            date = self._bar_dates.get(fill.fill_bar, pd.NaT)

            current = self.positions.get(
                symbol, {"position": Position.FLAT, "entry_price": 0.0, "size": 0.0}
            )

            if current["position"] == Position.LONG:
                # Sell-fill closes the long.
                assert order.side == -1, (
                    f"unexpected buy fill on LONG position for {symbol}"
                )
                old_entry = float(current["entry_price"])
                old_size = float(current["size"])
                pnl = (fill_price - old_entry) * old_size
                # Cash recovered: collateral (entry_price * size) + pnl - commission.
                # = fill_price * size - commission. Slippage already in fill_price.
                self.cash += fill_price * old_size - commission
                action = "CLOSE_LONG"
                self.positions[symbol] = {
                    "position": Position.FLAT, "entry_price": 0.0, "size": 0.0,
                }
                self._log_transaction(
                    date, symbol, action, fill_price, old_size,
                    commission, slippage_cost, pnl,
                )
                continue

            if current["position"] == Position.SHORT:
                # Buy-fill covers the short.
                assert order.side == 1, (
                    f"unexpected sell fill on SHORT position for {symbol}"
                )
                old_entry = float(current["entry_price"])
                old_size = float(current["size"])
                pnl = (old_entry - fill_price) * old_size
                # Long-style cash accounting: recover collateral (entry_price * size)
                # + short pnl - commission. = 2*entry*size - fill*size - commission.
                self.cash += old_entry * old_size + pnl - commission
                action = "CLOSE_SHORT"
                self.positions[symbol] = {
                    "position": Position.FLAT, "entry_price": 0.0, "size": 0.0,
                }
                self._log_transaction(
                    date, symbol, action, fill_price, old_size,
                    commission, slippage_cost, pnl,
                )
                continue

            # current is FLAT — this fill opens a new position.
            entry_notional = shares * fill_price
            # Long-style cash accounting: deduct entry collateral + commission for
            # both LONG and SHORT. update_portfolio_value's `2*entry - current`
            # short-value formula complements this convention.
            self.cash -= entry_notional + commission
            new_position = Position.LONG if order.side == 1 else Position.SHORT
            self.positions[symbol] = {
                "position": new_position, "entry_price": fill_price, "size": shares,
            }
            action = f"OPEN_{new_position.name}"
            self._log_transaction(
                date, symbol, action, fill_price, shares,
                commission, slippage_cost, 0.0,
            )

    def _log_transaction(
        self,
        date: pd.Timestamp,
        symbol: str,
        action: str,
        price: float,
        shares: float,
        commission: float,
        slippage_cost: float,
        pnl: float,
    ) -> None:
        net_pnl = pnl - commission - slippage_cost if pnl else -commission - slippage_cost
        self.transaction_log.append(
            {
                "date": date,
                "symbol": symbol,
                "action": action,
                "price": price,
                "shares": shares,
                "commission": commission,
                "slippage_cost": slippage_cost,
                "borrow_cost": 0.0,
                "pnl": pnl,
                "net_pnl": net_pnl,
                "cash": self.cash,
                "portfolio_value": self.portfolio_value,
            }
        )
        logger.info(
            f"{action} {symbol} {shares:.4f} @ {price:.2f}, "
            f"comm={commission:.4f}, slip={slippage_cost:.4f}, "
            f"pnl={pnl:.2f}, cash={self.cash:.2f}"
        )

    def accrue_borrow(
        self,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> float:
        """Charge per-bar borrow on every open short. Returns total $ charged.

        Borrow is computed on the mark-to-market notional (size * current_price),
        not entry notional — if the underlying rallies against a short, the
        borrow cost rises too.
        """
        total = 0.0
        exec_cfg = self.config.execution
        for symbol, info in self.positions.items():
            if info["position"] != Position.SHORT:
                continue
            price = prices.get(symbol)
            if price is None:
                continue
            size = float(info["size"])
            short_notional = size * float(price)
            charge = daily_borrow_dollars(
                short_notional=short_notional,
                borrow_rate_bps_annual=exec_cfg.borrow_rate_bps_annual,
                trading_days_per_year=exec_cfg.trading_days_per_year,
            )
            if charge <= 0:
                continue
            self.cash -= charge
            total += charge
            self.transaction_log.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": "BORROW",
                    "price": float(price),
                    "shares": size,
                    "commission": 0.0,
                    "slippage_cost": 0.0,
                    "borrow_cost": charge,
                    "pnl": 0.0,
                    "net_pnl": -charge,
                    "cash": self.cash,
                    "portfolio_value": self.portfolio_value,
                }
            )
        return total

    def apply_dividends(
        self,
        dividends: Optional[Dict[str, pd.Series]],
        date: pd.Timestamp,
    ) -> float:
        """Credit (long) / debit (short) cash dividends on the ex-date for any
        open position. Returns net cash applied (negative => net debit).

        Phase 4 §4.1: prices are split-adjusted but NOT dividend-adjusted, so a
        long held over an ex-date takes a markdown loss equal to the dividend
        in mark-to-market; crediting ``shares * amount`` makes the bar
        total-return neutral. A short pays the dividend (debit), mirroring
        borrow. ``dividends`` maps symbol -> ex-date Series (tz-aware ET);
        ``None``/missing is a strict no-op so non-dividend backtests are
        unaffected.
        """
        if not dividends:
            return 0.0
        total = 0.0
        for symbol, info in self.positions.items():
            if info["position"] == Position.FLAT:
                continue
            series = dividends.get(symbol)
            if series is None or len(series) == 0 or date not in series.index:
                continue
            amount = float(series.loc[date])
            if amount == 0.0:
                continue
            shares = float(info["size"])
            sign = 1.0 if info["position"] == Position.LONG else -1.0
            cash_delta = sign * shares * amount
            self.cash += cash_delta
            total += cash_delta
            self.transaction_log.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": "DIVIDEND",
                    "price": amount,
                    "shares": shares,
                    "commission": 0.0,
                    "slippage_cost": 0.0,
                    "borrow_cost": 0.0,
                    "pnl": cash_delta,
                    "net_pnl": cash_delta,
                    "cash": self.cash,
                    "portfolio_value": self.portfolio_value,
                }
            )
        return total

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

    def _reset_state(self) -> None:
        """Zero out per-backtest mutable state (positions, cash, history,
        transaction log, queued orders, bar-index map).

        Called once by ``backtest()`` and by WFO outer loops before their
        first segment. WFO loops must NOT call it between folds — state
        persists across folds and ``execution_model.drop_pending()`` is
        the only fold-boundary cleanup.
        """
        self.positions = {}
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.history = []
        self.transaction_log = []
        self.execution_model.drop_pending()
        self._bar_dates = {}

    def generate_target_weight_segment(
        self,
        data: Dict[str, pd.DataFrame],
        model_data: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sentiment: bool = True,
    ) -> pd.DataFrame:
        """Generate close-time target weights over one date segment.

        ``data`` remains the execution-price source and raw policy-member
        feature source. ``model_data`` optionally supplies fold-scaled features
        for forecast members, matching ``run_segment``'s train/serve replay
        path. The returned weights are close-time decisions; the execution
        layer is responsible for next-open fills and rebalance-band suppression.
        """
        if len(data) == 0:
            logger.warning("No data provided for target-weight segment")
            return pd.DataFrame()

        all_dates = pd.DatetimeIndex([])
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates = pd.DatetimeIndex(all_dates.union(df.index))
        all_dates = all_dates.sort_values()

        bar_tz = all_dates.tz if len(all_dates) else None
        if start_date:
            start_ts = pd.Timestamp(start_date)
            if bar_tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(bar_tz)
            all_dates = all_dates[all_dates >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            if bar_tz is not None and end_ts.tz is None:
                end_ts = end_ts.tz_localize(bar_tz)
            all_dates = all_dates[all_dates <= end_ts]

        if len(all_dates) == 0:
            logger.warning("No dates in the specified target-weight range")
            return pd.DataFrame()

        sentiment_data = {}
        if use_sentiment and self.use_sentiment and self.sentiment_analyzer:
            for symbol in data.keys():
                try:
                    sentiment_data[symbol] = self.sentiment_analyzer.create_sentiment_features(
                        symbol,
                        all_dates,
                    )
                except Exception as e:
                    logger.error(f"Error creating sentiment features for {symbol}: {e}")
                    sentiment_data[symbol] = pd.DataFrame(index=all_dates)

        rows: List[Dict[str, Union[pd.Timestamp, float]]] = []
        for date in all_dates:
            row: Dict[str, Union[pd.Timestamp, float]] = {"date": date}
            for symbol, symbol_data in data.items():
                if date not in symbol_data.index:
                    continue
                price_row = symbol_data.loc[date]
                price = float(price_row["close"])
                symbol_model_data = (
                    model_data.get(symbol, symbol_data)
                    if model_data is not None
                    else symbol_data
                )
                if date not in symbol_model_data.index:
                    logger.warning(
                        "No model-data row for %s on %s; skipping target",
                        symbol,
                        date,
                    )
                    continue

                hist = getattr(self.model, "required_history", 1)
                features_df = symbol_model_data.loc[:date].iloc[-hist:].copy()
                policy_features_df = symbol_data.loc[:date].iloc[-hist:].copy()
                label_cols = [
                    c for c in features_df.columns
                    if "target_" in c or "direction_" in c
                ]
                if label_cols:
                    features_df = features_df.drop(columns=label_cols)
                policy_label_cols = [
                    c for c in policy_features_df.columns
                    if "target_" in c or "direction_" in c
                ]
                if policy_label_cols:
                    policy_features_df = policy_features_df.drop(columns=policy_label_cols)

                symbol_sentiment = None
                if use_sentiment and symbol in sentiment_data:
                    symbol_sentiment = sentiment_data[symbol].loc[date:date].copy()

                try:
                    if isinstance(self.model, EnsembleModel):
                        predictions = self.model.predict(
                            features_df,
                            policy_X=policy_features_df,
                        )
                    else:
                        predictions = self.model.predict(features_df)
                    latest_prediction = predictions[-1:] if len(predictions) else predictions
                    row[symbol] = self.calculate_target_weight(
                        latest_prediction,
                        price,
                        symbol,
                        date,
                        symbol_sentiment,
                    )
                except Exception as e:
                    logger.error(f"Error generating target for {symbol} on {date}: {e}")
                    continue
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        weights = pd.DataFrame(rows).set_index("date")
        symbols = [symbol for symbol in data.keys() if symbol in weights.columns]
        weights = weights.reindex(columns=symbols).astype(float)
        return scale_to_max_gross(weights, self.config.max_gross_exposure)

    def run_segment(
        self,
        data: Dict[str, pd.DataFrame],
        model_data: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sentiment: bool = True,
        dividends: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Run the per-bar trading loop over one date segment without
        resetting state or dropping pending orders.

        The bar index continues from where the previous segment ended
        (``len(self._bar_dates)``) so cross-segment order/fill bookkeeping
        stays consistent. The caller decides when to ``_reset_state()`` and
        when to ``execution_model.drop_pending()`` — for a one-shot run use
        ``backtest()``; for a WFO loop, reset once before fold 0 and
        ``drop_pending()`` between folds.

        ``data`` remains the execution/accounting source. When ``model_data``
        is provided, prediction windows are drawn from it by symbol and date;
        this lets WFO replay use fold-scaled features without changing fill
        prices or marks.

        Returns a DataFrame indexed by date with ``portfolio_value`` and
        ``cash`` columns covering only the bars recorded in THIS segment
        (derived columns are computed by the caller after concatenation).
        """
        if len(data) == 0:
            logger.warning("No data provided for segment")
            return pd.DataFrame()

        start_history_pos = len(self.history)
        start_bar_idx = len(self._bar_dates)

        all_dates = pd.DatetimeIndex([])
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            all_dates = pd.DatetimeIndex(all_dates.union(df.index))
        all_dates = all_dates.sort_values()

        bar_tz = all_dates.tz if len(all_dates) else None
        if start_date:
            start_ts = pd.Timestamp(start_date)
            if bar_tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(bar_tz)
            all_dates = all_dates[all_dates >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            if bar_tz is not None and end_ts.tz is None:
                end_ts = end_ts.tz_localize(bar_tz)
            all_dates = all_dates[all_dates <= end_ts]

        if len(all_dates) == 0:
            logger.warning("No dates in the specified range")
            return pd.DataFrame()

        logger.info(f"Running segment from {all_dates[0]} to {all_dates[-1]}")

        sentiment_data = {}
        if use_sentiment and self.use_sentiment and self.sentiment_analyzer:
            for symbol in data.keys():
                try:
                    sentiment_features = self.sentiment_analyzer.create_sentiment_features(symbol, all_dates)
                    sentiment_data[symbol] = sentiment_features
                except Exception as e:
                    logger.error(f"Error creating sentiment features for {symbol}: {e}")
                    sentiment_data[symbol] = pd.DataFrame(index=all_dates)

        # Phase 2.5: per-symbol FIFO of conformal bands awaiting their
        # horizon-h realized ideal position. Local to this segment so ACI
        # adapts within a fold and resets cleanly across WFO folds (the model
        # itself re-inits self.aci on every refit). Entries are
        # (realize_pos, lower, upper, anchor_close, sigma_anchor) where
        # realize_pos is the symbol's own integer index position at t+h.
        conformal_pending: Dict[str, deque] = defaultdict(deque)

        for offset, date in enumerate(all_dates):
            bar_idx = start_bar_idx + offset
            self._bar_dates[bar_idx] = date

            bar_ohlc: Dict[str, Dict[str, float]] = {}
            current_prices: Dict[str, float] = {}
            for symbol, df in data.items():
                if date not in df.index:
                    continue
                row = df.loc[date]
                open_p = float(row["open"]) if "open" in df.columns else float(row["close"])
                close_p = float(row["close"])
                bar_ohlc[symbol] = {"open": open_p, "close": close_p}
                current_prices[symbol] = close_p

            if not current_prices:
                continue

            # 2. Fills from bar_idx-1.
            fills = self.execution_model.step(
                bar_idx=bar_idx,
                bar_ohlc=bar_ohlc,
                portfolio_value=self.portfolio_value,
            )
            if fills:
                self.apply_fills(fills)

            # 3. Mark-to-market.
            self.update_portfolio_value(current_prices)

            # 4. Generate signals + queue orders for bar_idx+1.
            for symbol, price in current_prices.items():
                symbol_data = data[symbol]
                symbol_model_data = (
                    model_data.get(symbol, symbol_data)
                    if model_data is not None
                    else symbol_data
                )
                if date not in symbol_model_data.index:
                    logger.warning(
                        "No model-data row for %s on %s; skipping signal",
                        symbol,
                        date,
                    )
                    continue
                # Feed the model enough trailing history to score this bar.
                # Point models need 1 row; windowed/sequence members (LSTM, RL
                # agents) need their lookback or predict() errors out and the
                # member contributes no signal. predict() returns one position
                # per fed row → the LAST element is this bar's decision.
                hist = getattr(self.model, "required_history", 1)
                features_df = symbol_model_data.loc[:date].iloc[-hist:].copy()
                policy_features_df = symbol_data.loc[:date].iloc[-hist:].copy()
                # Match the column set the members were FIT on: training drops
                # target_/direction_ label columns before ensemble.fit, and
                # XGBoost validates feature names strictly, so they must be
                # excluded here too (else every per-bar predict raises a
                # feature_names mismatch and the member contributes nothing).
                _label_cols = [
                    c for c in features_df.columns
                    if "target_" in c or "direction_" in c
                ]
                if _label_cols:
                    features_df = features_df.drop(columns=_label_cols)
                policy_label_cols = [
                    c for c in policy_features_df.columns
                    if "target_" in c or "direction_" in c
                ]
                if policy_label_cols:
                    policy_features_df = policy_features_df.drop(columns=policy_label_cols)

                conformal_on = (
                    isinstance(self.model, EnsembleModel) and self.model.has_conformal()
                )
                cur_pos = symbol_model_data.index.get_loc(date) if conformal_on else 0
                model_price = price
                if self.model.target_column in symbol_model_data.columns:
                    model_price = float(symbol_model_data.loc[date, self.model.target_column])

                # Phase 2.5: realize matured conformal bands → ACI online
                # update. A band queued at t is scored at t+h against the
                # realized ideal position (perfect-foresight bet on the actual
                # h-bar return), the same target the calibrator regressed on.
                if conformal_on:
                    pend = conformal_pending[symbol]
                    while pend and cur_pos >= pend[0][0]:
                        _, lo, up, anchor_close, sigma_a = pend.popleft()
                        realized_ret = model_price / anchor_close - 1.0
                        realized_ideal = float(
                            np.clip(
                                self.model.target_vol * realized_ret / sigma_a,
                                -self.model.position_cap,
                                self.model.position_cap,
                            )
                        )
                        self.model.update_aci(bool(lo <= realized_ideal <= up))

                symbol_sentiment = None
                if use_sentiment and symbol in sentiment_data:
                    symbol_sentiment = sentiment_data[symbol].loc[date:date].copy()

                try:
                    if isinstance(self.model, EnsembleModel):
                        predictions = self.model.predict(
                            features_df,
                            policy_X=policy_features_df,
                        )
                    else:
                        predictions = self.model.predict(features_df)
                    # predict() returns one position per fed row; the LAST row
                    # is this bar's decision (windowed members emit a vector).
                    latest_prediction = predictions[-1:] if len(predictions) else predictions
                    # Phase 2.5: compute conformal bands if available, and queue
                    # this band for ACI realization h bars ahead.
                    band_width = None
                    if conformal_on:
                        lower, point, upper = self.model.predict_band(
                            features_df,
                            policy_X=policy_features_df,
                        )
                        if len(lower) > 0 and len(upper) > 0:
                            band_width = float(upper[-1] - lower[-1])
                            sigma_now = float(
                                realized_vol(
                                    symbol_model_data[self.model.target_column].loc[:date],
                                    window=self.model.vol_window,
                                    vol_floor=self.model.vol_floor,
                                )[-1]
                            )
                            conformal_pending[symbol].append(
                                (
                                    cur_pos + self.model.horizon,
                                    float(lower[-1]),
                                    float(upper[-1]),
                                    model_price,
                                    sigma_now,
                                )
                            )
                    position, confidence = self.calculate_signal(
                        latest_prediction, price, symbol, date, symbol_sentiment,
                        band_width=band_width, features=features_df,
                        policy_features=policy_features_df,
                    )
                    self.submit_signal_order(
                        symbol=symbol,
                        position=position,
                        confidence=confidence,
                        signal_price=price,
                        bar_idx=bar_idx,
                    )
                except Exception as e:
                    logger.error(f"Error during segment for {symbol} on {date}: {e}")
                    continue

            # 5. Daily borrow on open shorts (mark-to-market).
            self.accrue_borrow(current_prices, date)

            # 5b. Cash dividends on ex-date for open positions (Phase 4 §4.1).
            self.apply_dividends(dividends, date)

            # 6. Final mark + record.
            portfolio_value = self.update_portfolio_value(current_prices)
            self.history.append(
                {"date": date, "portfolio_value": portfolio_value, "cash": self.cash}
            )

        segment_rows = self.history[start_history_pos:]
        if not segment_rows:
            return pd.DataFrame()
        return pd.DataFrame(segment_rows).set_index("date")

    @staticmethod
    def _finalize_results(seg_df: pd.DataFrame) -> pd.DataFrame:
        """Append derived columns (daily_return, cumulative_return, drawdown)
        to a history DataFrame. Idempotent on the input frame (returns a copy).

        Called once on the full ``backtest()`` output, or once on the
        concatenated multi-fold WFO frame so the equity curve reads as one
        continuous portfolio rather than per-fold resets.
        """
        result = seg_df.copy()
        result["daily_return"] = result["portfolio_value"].pct_change()
        result["cumulative_return"] = (1 + result["daily_return"]).cumprod() - 1
        result["drawdown"] = 1 - result["portfolio_value"] / result["portfolio_value"].cummax()
        return result

    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sentiment: bool = True,
        dividends: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Run a one-shot backtest with ExecutionModel-managed fills + costs.

        Per-bar timeline (bar_idx t, date D_t):
          1. Fetch OHLC for every symbol with data on D_t.
          2. Apply ExecutionModel.step(t, …) — fills any orders submitted at
             bar t-1 (MOO at open(t), MOC at close(t)). No same-bar fill.
          3. Mark-to-market portfolio at close(t).
          4. For each symbol: predict, derive (position, confidence),
             submit_signal_order(t) — orders queue for bar t+1.
          5. accrue_borrow on every open short (mark-to-market at close).
          6. Mark-to-market again post-borrow, record history.

        Orders queued at the last bar of `data` never fill — they are dropped
        at the end of the loop, mirroring fold-boundary behavior.

        Composition of ``_reset_state`` → ``run_segment`` → ``drop_pending``
        → ``_finalize_results``. WFO outer loops call those primitives
        directly and skip this wrapper.
        """
        self._reset_state()
        seg_df = self.run_segment(
            data=data,
            start_date=start_date,
            end_date=end_date,
            use_sentiment=use_sentiment,
            dividends=dividends,
        )
        self.execution_model.drop_pending()
        if seg_df.empty:
            return seg_df
        return self._finalize_results(seg_df)

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

        # Sharpe ratio (annualized; the canonical reporting number)
        if metrics["volatility"] > 0:
            excess_return = metrics["annualized_return"] - self.config.risk_free_rate
            metrics["sharpe_ratio"] = excess_return / metrics["volatility"]

        # Probabilistic Sharpe Ratio (Bailey/LdP 2012): probability that the
        # true periodic Sharpe exceeds 0 given the observed sample's higher
        # moments. Computed on daily returns; benchmark 0 i.e. "any edge".
        daily_returns = results["daily_return"].dropna().to_numpy()
        metrics["psr"] = probabilistic_sharpe_ratio(daily_returns, sr_benchmark=0.0)

        # Maximum drawdown (positive convention: 1 - portfolio/cummax ≥ 0).
        metrics["max_drawdown"] = results["drawdown"].max()

        # Calmar ratio — NaN when max_drawdown is non-positive.
        metrics["calmar_ratio"] = calmar_ratio(
            metrics.get("annualized_return", float("nan")),
            metrics["max_drawdown"],
        )

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
        report.append(f"Sharpe ratio: {metrics.get('sharpe_ratio', float('nan')):.2f}")
        psr = metrics.get("psr", float("nan"))
        report.append(f"Probabilistic Sharpe (P[SR>0]): {psr:.3f}" if np.isfinite(psr) else "Probabilistic Sharpe (P[SR>0]): n/a")
        report.append(f"Maximum drawdown: {metrics.get('max_drawdown', 0):.2%}")
        calmar = metrics.get("calmar_ratio", float("nan"))
        report.append(f"Calmar ratio: {calmar:.2f}" if np.isfinite(calmar) else "Calmar ratio: n/a")
        report.append("")

        # Trading statistics
        report.append("-" * 50)
        report.append("TRADING STATISTICS")
        report.append("-" * 50)

        if self.transaction_log:
            trades_df = pd.DataFrame(self.transaction_log)
            opens = trades_df[trades_df["action"].str.startswith("OPEN")]

            report.append(f"Total trades: {len(opens)}")

            if "win_rate" in metrics:
                report.append(f"Win rate: {metrics['win_rate']:.2%}")
                report.append(f"Average profit: ${metrics.get('avg_profit', 0):.2f}")
                report.append(f"Average loss: ${metrics.get('avg_loss', 0):.2f}")
                report.append(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")

            # Calculate by position type
            long_opens = trades_df[trades_df["action"] == "OPEN_LONG"]
            short_opens = trades_df[trades_df["action"] == "OPEN_SHORT"]

            report.append(f"Long trades: {len(long_opens)}")
            report.append(f"Short trades: {len(short_opens)}")

            # Cost breakdown (commission + slippage + borrow). Older transaction
            # log rows may lack the new columns when loaded from a stale pickle.
            total_commission = trades_df["commission"].sum() if "commission" in trades_df else 0.0
            total_slippage = trades_df["slippage_cost"].sum() if "slippage_cost" in trades_df else 0.0
            total_borrow = trades_df["borrow_cost"].sum() if "borrow_cost" in trades_df else 0.0
            report.append(f"Total commission: ${total_commission:.2f}")
            report.append(f"Total slippage:   ${total_slippage:.2f}")
            report.append(f"Total borrow:     ${total_borrow:.2f}")
            report.append(
                f"Total costs:      ${total_commission + total_slippage + total_borrow:.2f}"
            )

        report.append("")
        report.append("=" * 50)

        return "\n".join(report)
