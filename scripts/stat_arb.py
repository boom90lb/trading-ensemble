"""Train-only cointegration pair discovery and costed stat-arb backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.arbitrage import (
    PairSelectionConfig,
    PairSignalConfig,
    backtest_target_weights,
    combine_pair_positions,
    generate_pair_positions,
    scan_cointegrated_pairs,
)
from src.config import ExecutionConfig, RESULTS_DIR
from src.data_loader import DataLoader
from src.logging_utils import configure_logging, get_symbol_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a train-only pairs stat-arb research backtest.")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols.")
    p.add_argument("--start_date", type=str, default="2020-01-01")
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--formation_bars", type=int, default=504)
    p.add_argument("--max_pairs", type=int, default=10)
    p.add_argument("--fdr_alpha", type=float, default=0.10)
    p.add_argument("--entry_z", type=float, default=2.0)
    p.add_argument("--exit_z", type=float, default=0.5)
    p.add_argument("--stop_z", type=float, default=4.0)
    p.add_argument("--z_window", type=int, default=60)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--max_symbol_abs_weight", type=float, default=0.35)
    p.add_argument("--commission_bps", type=float, default=1.0)
    p.add_argument("--spread_bps", type=float, default=1.0)
    p.add_argument("--slippage_coeff", type=float, default=10.0)
    p.add_argument("--borrow_bps_annual", type=float, default=50.0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _fetch_matrices(symbols: List[str], start_date: str, end_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    close_matrix, open_matrix = _fetch_matrices(symbols, args.start_date, args.end_date)
    if len(close_matrix) <= args.formation_bars + 2:
        raise RuntimeError(
            f"Need more than formation_bars+2 rows; got {len(close_matrix)} rows and "
            f"formation_bars={args.formation_bars}"
        )

    formation_close = close_matrix.iloc[: args.formation_bars]
    selection_cfg = PairSelectionConfig(
        min_obs=min(args.formation_bars, 252),
        max_pairs=args.max_pairs,
        fdr_alpha=args.fdr_alpha,
    )
    signal_cfg = PairSignalConfig(
        z_window=args.z_window,
        min_z_obs=max(2, min(args.z_window, args.z_window // 2)),
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
    )
    candidates = scan_cointegrated_pairs(formation_close, selection_cfg)
    signal_frames = [generate_pair_positions(close_matrix, c, signal_cfg) for c in candidates]
    combined = combine_pair_positions(
        (sf.target_weights for sf in signal_frames),
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_abs_weight,
    )
    if combined.empty:
        raise RuntimeError("No pairs survived selection; no portfolio to backtest.")

    # Only evaluate the post-formation period. Formation rows are zeroed so pair
    # discovery never receives PnL credit for the same sample used for selection.
    combined.loc[combined.index[: args.formation_bars], :] = 0.0
    execution = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        commission_bps=args.commission_bps,
        borrow_rate_bps_annual=args.borrow_bps_annual,
    )
    result = backtest_target_weights(open_matrix, combined, execution=execution)

    out_dir = (
        Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"stat_arb_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_payload = [c.__dict__ for c in candidates]
    (out_dir / "pairs.json").write_text(json.dumps(pairs_payload, indent=2))
    result.returns.to_csv(out_dir / "returns.csv")
    result.equity.to_csv(out_dir / "equity.csv")
    result.target_weights.to_csv(out_dir / "target_weights.csv")
    result.costs.to_csv(out_dir / "costs.csv")
    (out_dir / "summary.json").write_text(json.dumps(result.metrics, indent=2, allow_nan=True))
    print(
        json.dumps(
            {"output_dir": str(out_dir), "n_pairs": len(candidates), **result.metrics},
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
