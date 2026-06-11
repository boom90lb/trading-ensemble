# trading-ensemble

A research backtesting stack that combines classical time-series forecasters
(ARIMA, Prophet, LSTM, XGBoost) and three reinforcement-learning policies
(LSTM-PPO, xLSTM-PPO, xLSTM-GRPO) into a single position-space ensemble, then
evaluates it under a **methodology designed to produce honest out-of-sample
numbers** — purged walk-forward CV, next-open fills with realistic costs,
mandatory baselines, and overfitting-adjusted metrics (DSR, PBO).

It also includes a separate statistical-arbitrage path for market-neutral pair
research: train-only cointegration discovery, residual stationarity and
multiple-testing filters, causal spread targets, capped portfolio weights, and
next-open costed accounting. See [`docs/stat_arb.md`](docs/stat_arb.md).

The emphasis is on *trustworthy evaluation*, not on a deployable alpha. The
default universe and hyperparameters are illustrative; the value is in the
harness around them.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the end-to-end data flow (what
calls what) and [`docs/operations.md`](docs/operations.md) for operational
gotchas (vendor tier, interval mapping, member contribution, per-bar cost).

> **Note on the RL members.** The three RL agents were stub implementations in
> the original codebase (hardcoded losses, random-action `predict`). They now
> have real gradient updates and end-to-end fit/predict. If you extend them,
> check that the policy actually takes gradient steps and that a windowed
> member produces non-flat positions before trusting any RL-attributed Sharpe
> — under a small training budget an undertrained policy legitimately produces
> ~zero positions and a NaN Sharpe. See `scripts/rl_seed_eval.py` for the
> multi-seed overfitting study.

---

## Methodology

Every choice below exists because the naive alternative makes a backtest look
better than the strategy is. Read these before interpreting any number.

### Purged, embargoed walk-forward CV (López de Prado, AFML §7.4)
`src/validation/walk_forward.py:PurgedWalkForward` drives both training and
backtest. Two distinct leakage controls:

- **Purge** — training rows whose forward-label window overlaps the test slice
  are dropped (`purge_horizon`, defaulting to the prediction horizon, since the
  label is a horizon-bar forward return).
- **Embargo** — a buffer after each test slice is excluded from *subsequent*
  folds (`embargo_pct`).

Both `scripts/training.py` and `scripts/backtest.py` iterate the **same** fold
structure: training writes `split_idx.npz` per fold; the backtest replays the
identical test-date ranges. There is no 80/20 split anywhere.

### Execution model: signal on close, fill at next open
`src/execution/execution_model.py`. A signal computed on bar *t*'s close is
filled at bar *t+1* (market-on-open by default, `--order_type MOC` for
market-on-close). Nothing fills same-bar. Costs applied per fill:

- half-spread (`--spread_bps`) + linear notional impact (`--slippage_coeff`),
- commission in bps (`--commission_bps`),
- daily **borrow** on open short notional (`--borrow_bps_annual`) — shorts are
  not free.

Reported PnL is **net** of all of the above on a fold-aligned, concatenated
equity curve.

### Baselines are mandatory
`src/baselines/`: Buy-and-Hold, MA-crossover (`--ma_fast`/`--ma_slow`), and
time-series momentum (`--tsmom_lookback`). They traverse the *same* fold
structure with the same costs, so the comparison is fair and the cross-strategy
PBO is well-defined. Run with `--no_baselines` to skip (not recommended).

### Overfitting-adjusted metrics
`src/validation/metrics.py`:

- **PSR** (Probabilistic Sharpe Ratio) — skew/kurtosis-adjusted probability the
  true Sharpe exceeds 0.
- **PBO** (Probability of Backtest Overfitting) via CSCV — across the
  {ensemble + baselines} strategy set, the fraction of IS/OOS splits where the
  IS-best strategy underperforms OOS. High PBO ⇒ the selection is overfit.
- **DSR** (Deflated Sharpe Ratio) — PSR with the benchmark set to the *expected
  maximum* Sharpe under the False Strategy Theorem given the number of trials.
  Computed by `scripts/sweep.py` over a real hyperparameter grid; the backtest
  shows it when you pass `--trial_sharpes_json`.

### Conformal prediction bands (EnbPI + ACI)
`src/conformal/`. The ensemble emits a position band, not just a point. EnbPI
reuses the meta-learner's out-of-fold residuals for finite-sample-valid
intervals (block-cross-conformal, **not** split conformal — the latter assumes
exchangeability that time series violate). ACI (Gibbs & Candès) adapts the
miscoverage level α online as outcomes realize. Wide (uncertain) bands dampen
position size in `trading.calculate_signal`.

### Leakage audit
Done before relying on any WFO number (a WFO over leaky features is a
well-organized lie). Closed leaks: point-in-time UTC sentiment bucketing
(`searchsorted` against bar-close times, no across-bar ffill), train-only
feature clipping bounds, per-fold scaler refits. See `src/sentiment_analysis.py`
and `src/features.py` plus `tests/test_sentiment_leakage.py` /
`tests/test_feature_engineer_leakage.py`.

---

## How to read the backtest report

`scripts/backtest.py` prints a per-strategy comparison block:

```
Strategy        TotRet    AnnRet   Sharpe  PSR(>0)    MaxDD   Calmar
```

- **TotRet / AnnRet** — total / annualized return, net of costs.
- **Sharpe** — annualized; a headline number, *not* the one to trust alone.
- **PSR(>0)** — probability the true Sharpe is positive after skew/kurtosis
  adjustment. Treat a Sharpe with low PSR as noise.
- **MaxDD** — maximum drawdown (positive-magnitude convention).
- **Calmar** — annualized return / max drawdown; `n/a` when undefined.

Footers (one per backtest, not per strategy):

- **PBO** — across all strategies in the table. The single most important line:
  a strong ensemble Sharpe with PBO near or above 0.5 means the result is
  likely selection-overfit.
- **Deflated Sharpe Ratio** — shown when `--trial_sharpes_json` from a sweep is
  supplied; the ensemble Sharpe deflated by the trial count. **Read DSR, not
  raw Sharpe, when a grid was searched.**

Per-strategy returns are inner-joined by date before PBO so the matrix is
fold-aligned.

---

## Usage

Environment (Python 3.12+, [uv](https://github.com/astral-sh/uv)):

```bash
git clone https://github.com/boom90lb/trading-ensemble.git
cd trading-ensemble
uv venv && uv pip install -e .
```

API keys in a `.env` (Twelvedata for bars/dividends; Polygon for sentiment):

```
TWELVEDATA_API_KEY=...
POLYGON_API_KEY=...
```

### 1. Train (produces a run directory the backtest replays)

```bash
# Per-symbol purged WFO; writes runs/{run_name}/{symbol}/fold_*/.
python -m scripts.training --symbols AAPL,MSFT,GOOG --start_date 2018-01-01 \
    --horizon 5 --n_splits 5

# Universe file + as-of point-in-time inclusion guard; with sentiment.
python -m scripts.training --universe data/universe/2026-05-30.txt \
    --universe_asof 2018-01-01 --use_sentiment

# RL members at a real training budget (the default 100k is heavy).
python -m scripts.training --symbols AAPL --models xgboost,lstm_ppo \
    --rl_timesteps 200000
```

### 2. Backtest (requires a training run; never a bare symbol list)

```bash
python -m scripts.backtest --training_run runs/{run_name}

# With a sweep's trial Sharpes so the report shows DSR:
python -m scripts.backtest --training_run runs/{run_name} \
    --trial_sharpes_json results/{sweep}/trial_sharpes.json
```

### 3. Hyperparameter sweep → honest DSR (forecast-only)

```bash
python -m scripts.sweep --symbols AAPL,MSFT
# writes selected_config.json + trial_sharpes.json (feed the latter to backtest)
```

### 4. Multi-seed RL overfitting study

```bash
# EXPENSIVE: |members| x |symbols| x |folds| x |seeds| bare RL fits.
python -m scripts.rl_seed_eval --training_run runs/{run_name} \
    --members lstm_ppo --seeds 0,1,2 --rl_timesteps 50000
```

### 5. Statistical arbitrage pair research

```bash
# Rolling formation/test walk-forward. This is the credible research path.
python -m scripts.stat_arb_wfo --symbols AAPL,MSFT,GOOG,AMZN,META,NVDA \
    --start_date 2020-01-01 --formation_bars 504 --test_bars 63 --max_pairs 5

# Single formation/test smoke test.
python -m scripts.stat_arb --symbols AAPL,MSFT,GOOG,AMZN,META,NVDA \
    --start_date 2020-01-01 --formation_bars 504 --max_pairs 5
```

This is the only path in the repo that should currently be called
"arbitrage-like": it forms cross-asset hedge-ratio books rather than independent
single-symbol bets. The WFO command writes fold-level pair selection ledgers,
target weights, returns, costs, pair-trial Sharpes, and a `pair_set_dsr` field
so pair-search bias is visible rather than hidden behind a raw Sharpe.

Add `--verbose` to any script for console DEBUG output (the per-run log file is
always DEBUG regardless — see **Logging**).

---

## Logging

`src/logging_utils.configure_logging` (called by every script's `main`)
installs:

- a **console** handler at INFO (DEBUG with `--verbose`), and
- a **rotating file** handler at `logs/run_{ts}.log` that always captures DEBUG
  (50 MB cap, 5 backups) — the file is the complete record even when the
  console is quiet.

Per-symbol log records carry `extra={"symbol": ...}` (via `get_symbol_logger`),
so the file is greppable by ticker and aligns with the per-symbol MLflow runs.
`logs/` is gitignored.

MLflow tracks a parent run per invocation, nested per symbol, with per-fold
metrics at `step=fold_idx`. `MLFLOW_TRACKING_URI` defaults to a local
`file://.../mlruns` store; `sqlite:///mlflow.db` is the documented migration
target (the file backend is deprecated).

---

## Data quality

**Corporate actions.** Prices are fetched `adjust=splits` — split-adjusted but
**not** dividend-adjusted, so the close is a faithful tradeable price.
Dividends are credited as **explicit cash** in the backtest
(`DataLoader.fetch_dividends` → `TradingStrategy.apply_dividends`): a long held
over an ex-date takes a mark-to-market markdown that the per-share credit
offsets (a short is debited), making the position total-return correct without
back-adjusting prices. Trade-off: ex-dividend gaps remain in the
return/volatility *features*. `--no_dividends` disables the credit.

**Cache integrity.** Cached bars are keyed by requested range
(`{symbol}_{interval}_{start}_{end}.parquet`) and reused only when the cached
range *contains* the request, so a narrow cache can't silently satisfy a wider
query with a too-short slice.

**Operational gotchas** (vendor tier limits, the `1d`→`1day` interval mapping,
which ensemble members actually feed the backtest, and the per-bar JAX cost)
are documented in [`docs/operations.md`](docs/operations.md).

**Universe & survivorship.** `--universe <file>` (one symbol per line, `#`
comments) and `--universe_asof YYYY-MM-DD` (drop names with no data
at/before the date) give a **best-effort** point-in-time universe on the
*included* names. It does **not** recover delisted/acquired tickers — true
survivorship-bias-free construction needs a delisting database (CRSP, Norgate),
which is **out of scope**. Read results with that residual survivorship bias in
mind.

---

## Out of scope

- Point-in-time universe via a delisting database (CRSP/Norgate).
- Sentiment distillation. The pipeline ships keyword `SentimentAnalyzer` and an
  interim FinBERT `TransformerSentimentAnalyzer`; the white-box distillation
  plan is documented in `docs/sentiment_roadmap.md` but not implemented.
- Cross-symbol shared-cash portfolio (each symbol is backtested independently).
- A market simulator for multi-step synthetic OHLCV; positions are held flat
  across the prediction horizon rather than iteratively re-forecast.

---

## Project layout

```
src/
  config.py            single source of truth for weights / hyperparams / dirs
  data_loader.py       Twelvedata bars + dividends, range-keyed cache
  features.py          technical indicators, train-only clip bounds
  sentiment_analysis.py keyword + FinBERT analyzers, PIT bucketing
  trading.py           signals, position accounting, backtest loop
  logging_utils.py     configure_logging + per-symbol adapter (Phase 5.1)
  models/              arima, prophet, lstm, xgboost, {lstm,xlstm}_ppo, xlstm_grpo,
                       ensemble, registry (forecast vs policy), mapping (vol-sizing)
  baselines/           buy-and-hold, MA-crossover, TSMOM
  execution/           ExecutionModel + cost functions
  validation/          PurgedWalkForward, metrics (PSR/PBO/DSR/Calmar)
  conformal/           EnbPI + ACI
  arbitrage/           cointegration pair scan, causal spread signals,
                       capped portfolio accounting
  tracking/            MLflow wrappers
scripts/
  training.py          per-symbol purged-WFO training → runs/{run_name}/
  backtest.py          WFO backtest vs per-fold models + baselines + PBO/DSR
  sweep.py             ensemble-layer grid → DSR
  rl_seed_eval.py      multi-seed RL overfitting study
  stat_arb.py          train-only pairs stat-arb backtest
  stat_arb_wfo.py      rolling formation/test pairs stat-arb WFO
  prediction.py        point predictions from a saved model
tests/                 ~223 tests (validation, leakage, execution, conformal, logging)
```

## Configuration

`src/config.py` is the single source of truth.

- `DEFAULT_MODEL_WEIGHTS` registers every ensemble member; scripts read weights
  from here rather than hardcoding per-model overrides.
- `TrainingConfig`/`ExecutionConfig`/`TradingConfig` validate their fields in
  `__post_init__` (e.g. `n_splits >= 2`, `0 <= embargo_pct < 1`,
  `0 < position_size <= 1`, `borrow_rate_bps_annual >= 0`). Invalid CLI
  arguments fail fast at config construction.
- `resolve_jax_device()` maps `"auto"` to gpu/tpu/cpu from `jax.devices()`.

---

## Dependencies

Python 3.12+ · scikit-learn, Flax/JAX, PyTorch · Gymnasium · statsmodels,
Prophet, XGBoost · transformers (FinBERT) · MLflow · Twelvedata, Polygon.io.

## License

MIT License, Copyright (c) 2026 Brendon Reperttang
