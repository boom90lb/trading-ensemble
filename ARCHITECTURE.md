# Architecture

How the pieces connect, end to end. The README covers *why* (methodology) and
*how to run* (usage); this is the *what calls what* map. `docs/operations.md`
covers operational gotchas.

## Data flow

```
Twelvedata API ──► DataLoader ──► FeatureEngineer ──► PurgedWalkForward ──► EnsembleModel ──► TradingStrategy ──► target-weight engine ──► claim packet
   (1day bars,      (range-keyed     (technical         (AFML §7.4          (forecast +        (close-time        (next-open fills,
    adjust=splits)   parquet cache)   indicators,        purge+embargo)      policy members,    target weights)    costs, borrow)
                                      train-only clip)                       vol-sized blend)
```

The default directional WFO backtest uses `TradingStrategy` only to generate
continuous close-time target weights from per-fold ensemble models. Portfolio
accounting then happens once across all symbols in `src/execution/target_weights.py`.
The legacy `--legacy_orders` path still uses `TradingStrategy.run_segment` and
`ExecutionModel` directly for per-symbol LONG/SHORT/FLAT order accounting.

The statistical-arbitrage path is intentionally separate:

```
close/open matrix ──► rolling formation/test folds ──► train-only pair scan ──► causal spread targets ──► capped pair portfolio ──► next-open accounting
                    (fixed candidates per fold)       (coint + ADF + FDR)      (z-score state machine)    (gross/symbol limits)     (costs + borrow)
```

### 1. Ingestion — `src/data_loader.py`
- `DataLoader.fetch_historical_data` pulls split-adjusted daily bars, caches to
  `data/{symbol}_{interval}_{start}_{end}.parquet`, returns a tz-aware
  (America/New_York) OHLCV frame. Interval is normalized to the vendor's
  `1day`/`1week`/`1month` only at the request boundary.
- `fetch_dividends` pulls ex-date→amount cash dividends (credited in the
  backtest, not back-adjusted into prices).

### 2. Features — `src/features.py`
- `create_features` (causal technical indicators) → `create_lagged_features` →
  `create_target_variable` (forward return `target_{h}`). Computed once on the
  full series (rolling/ewm/pct_change are causal).
- `fit_scalers` / `transform_features` apply **train-only** clip bounds —
  refit per WFO fold so test-fold outliers never leak into the scaler.

### 3. Splitting — `src/validation/walk_forward.py`
- `PurgedWalkForward` yields expanding (or rolling) train/test index pairs with
  a **purge** (drop train rows whose label window overlaps test) and an
  **embargo** (skip a buffer after each test). One splitter drives both the
  outer training loop and the meta-learner's OOF.

### 4. Models — `src/models/`
- `registry.py` partitions members into **forecast** (`arima`, `prophet`,
  `lstm`, `xgboost` — emit ŷ price) and **policy** (`lstm_ppo`, `xlstm_ppo`,
  `xlstm_grpo` — emit positions). `base.py` is the shared interface;
  `required_history` tells the backtest how many trailing bars each member needs.
- `mapping.py` bridges the two: `forecast_to_position` vol-sizes a price
  forecast into a position ∈ [−1, 1]; `ideal_position` is the perfect-foresight
  target the meta-learner regresses against.
- `ensemble.py` is the orchestrator:
  - `fit` fits each member, then `_fit_meta_learner_weights` solves a
    constrained NNLS (SLSQP, w≥0, Σw=1) over **OOF** positions vs. the ideal
    target, and fits the conformal calibrator on the same residuals.
  - `predict` routes each member through the forecast→position or clip path and
    returns the weighted blend in position space.
  - `save`/`load` persist ensemble metadata **and delegate to each member's own
    save/load** (forecast members write per-member subdirs; RL members store a
    path). A loaded ensemble must come back with every member `is_fitted`.
  - `predict_band` / `update_aci` provide EnbPI+ACI conformal bands.

### 5. Execution + accounting — `src/execution/`, `src/trading.py`
- `target_weights.py` is the default portfolio accounting path. It accepts
  close-time target weights, fills them at the next open, suppresses small
  rebalances by weight band, drops fold-last pending targets, carries existing
  filled weights across folds, row-scales gross exposure, applies spread,
  commission, impact, borrow, and dividend return contribution, and emits
  `target_weights`, `fill_weights`, `costs`, returns, equity, and metrics.
- `TradingStrategy.generate_target_weight_segment` is the default WFO signal
  surface: it replays each fold's saved feature state, feeds forecast members
  fold-scaled features and policy members raw OHLCV windows, and converts the
  ensemble position to `position_size * position`.
- `ExecutionModel` remains the legacy order engine. It queues `Order`s and fills
  them at **t+1** (MOO at open, MOC at close — never same-bar), applying
  half-spread + linear impact slippage, bps commission, and daily borrow on
  shorts (`costs.py`).
- `TradingStrategy.run_segment` is the legacy per-bar loop: predict →
  `calculate_signal` (position sign → LONG/SHORT/FLAT; magnitude +
  inter-model agreement + conformal band width → confidence) →
  `submit_signal_order` → next-bar fill → mark-to-market → borrow → record.
  `backtest` = `_reset_state → run_segment → drop_pending → _finalize_results`.

### 6. Evaluation — `src/validation/metrics.py`, `src/baselines/`
- `metrics.py`: PSR, PBO (CSCV), DSR (False Strategy Theorem), Calmar.
- `validation/trials.py`: canonical research claim packets with config hash,
  code commit, data convention, artifact manifest, gross/net/cost metrics,
  DSR/trial count when available, and claim tier.
- `baselines/`: BuyAndHold, MACrossover, TSMOM — used by the legacy order-mode
  comparison path so PBO is computed across a fold-aligned strategy set.
- `conformal/`: EnbPI block-cross-conformal + ACI online α-adaptation.

### 7. Statistical arbitrage — `src/arbitrage/`, `scripts/stat_arb*.py`
- `pairs.py`: train-only Engle-Granger pair scan, residual ADF check,
  Benjamini-Hochberg FDR control, half-life and beta-drift filters, and causal
  spread-z target generation. The report variant also records raw candidates,
  the FDR cutoff, and rejection counts.
- `portfolio.py`: compatibility exports for the shared target-weight portfolio
  accounting in `src/execution/target_weights.py`.
- `walk_forward.py`: rolls formation/test windows, freezes selected pairs per
  fold, forces fold-end targets flat, records pair turnover and fold metrics,
  and reports `pair_set_dsr` from selected pair-book trial Sharpes.
- `factors.py` (residual mode): causal eligibility (history/price/dollar-volume
  floors), weekly standardized-return PCA eigenportfolios (`Q = v/sigma`,
  sign-fixed), and per-day batched OLS of stock returns on factor returns.
- `residual.py`: AR(1)/OU fits on cumulative residuals, drift-adjusted A-L
  s-scores (invalid/slow fits are counted, never traded), the threshold state
  machine, and per-symbol netting of stock + eigenportfolio hedge legs into
  close-time targets.
- `residual_walk_forward.py`: same fold geometry and flattening as pairs, but
  estimators re-roll causally through test bars (nothing is frozen per fold);
  reports per-fold names traded, gross, cost share, and invalid-OU rates.
  The residual overfit control is the cross-run trial ledger
  (`results/stat_arb_residual_trials.jsonl`, written by the CLI), which feeds
  `residual_set_dsr`.
- This is the market-neutral research path. It does not use the single-symbol
  ensemble, because independent ticker forecasts are not an arbitrage book.

## Scripts (`scripts/`) — the entry points

| Script | Role | Key output |
|---|---|---|
| `training.py` | per-symbol purged-WFO fit | `runs/{run}/{symbol}/fold_*/ensemble_model/` + `split_idx.npz` |
| `backtest.py` | default shared-capital target-weight WFO; optional legacy order-mode baselines + PBO/DSR | root `claim_packet.json`, target/fill/cost/equity CSVs, `summary.json` |
| `sweep.py` | ensemble-layer grid → honest DSR (forecast-only) | `trial_sharpes.json`, `selected_config.json` |
| `rl_seed_eval.py` | multi-seed RL overfitting study | `rl_seed_eval.json` |
| `stat_arb_wfo.py` | rolling formation/test stat-arb WFO with fold-selection ledgers | `folds.json`, `pairs.json`, `summary.json`, weights/costs CSVs |
| `stat_arb_residual_wfo.py` | residual (Avellaneda-Lee) WFO over a universe file + cross-run trial ledger | `folds.json`, `summary.json`, `config.json`, weights/costs CSVs, `stat_arb_residual_trials.jsonl` |
| `prediction.py` | point predictions from a saved model | CSV + plot |

`config.py` is the single source of truth (weights, hyperparams, dirs, dataclass
configs). `tracking/mlflow_utils.py` wraps MLflow logging. `logging_utils.py`
installs console+rotating-file logging.

## Cross-cutting invariants worth knowing

- **Bars are tz-aware (ET)** from ingestion onward; `clean_data_for_training`
  asserts it. Prophet strips the zone internally because it requires tz-naive `ds`.
- **Label columns (`target_`/`direction_`) are dropped before fit AND before
  predict.** The training loop and `run_segment` must agree on the column set or
  XGBoost raises a feature-name mismatch.
- **The ensemble output is a position, not a price** — forecast members are
  mapped before blending (the B8 fix); never average ŷ with positions.
- **Members can be silently dropped from the blend** if `predict` returns a
  length ≠ `len(X)` — currently the forecast `lstm`. See `docs/operations.md`.
- **Directional WFO is now portfolio-level by default.** A strategy claiming a
  directional ensemble result should point to the root target/fill/cost/equity
  artifacts and `claim_packet.json`, not only per-symbol logs.
- **Arbitrage logic lives in `src/arbitrage/`**, not in the directional
  ensemble. A strategy claiming arbitrage should produce cross-asset hedged
  target weights and portfolio-level PnL, not independent symbol signals.
