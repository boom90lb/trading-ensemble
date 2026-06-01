# Operations notes

Durable operational facts for running the pipeline on real data. Methodology
lives in the README; this file is the "what will actually bite you" companion.

## Data vendor (Twelvedata)

- **API key** goes in a gitignored `.env` at the repo root:
  `TWELVEDATA_API_KEY=...`. `src/config.py` loads it via `python-dotenv`.
- **Interval strings.** The vendor's `time_series` endpoint expects
  `1day`/`1week`/`1month` (not `1d`/`1wk`/`1mo`). The project keeps the short
  form everywhere (config, cache filenames, CLI) and normalizes to the vendor
  spelling only at the request boundary (`DataLoader._to_vendor_interval`).
  Intraday strings (`1h`, `2h`, …) already match and pass through.
- **Tier matters.** On the `basic` tier:
  - Rate limit is ~8 requests/min, 800/day. Pre-warm the cache (one fetch per
    symbol) before a multi-symbol run; subsequent reads hit the parquet cache.
  - The `/dividends` endpoint is **403 Forbidden** for most symbols (AAPL has
    slipped through). Cross-symbol total-return is therefore inconsistent on
    this tier — run backtests with `--no_dividends` for a uniform price-return
    comparison, or supply a higher-tier key.
  - **Dividend-cache poisoning guard:** a 403/empty response caches an empty
    dividend Series, which the loader treats as authoritative "no dividends"
    and never refetches. If you upgrade tiers, delete the stale empty
    `data/*_dividends_*.parquet` files so they refetch.

## Which ensemble members actually contribute in a backtest

Not every configured member feeds the backtest blend:

- **arima, prophet, xgboost** — contribute. (Prophet needs a tz-naive `ds`;
  the model strips the zone internally, but that's why the bars being
  tz-localized matters.)
- **forecast `lstm`** — currently **dropped** in the backtest blend.
  `EnsembleModel._predict_positions_per_model` discards any member whose
  `predict` returns `len(raw) != len(X)`, and the forecast LSTM returns
  `len - sequence_length + 1`. Until that length reconciliation is built, an
  ensemble's forecast-LSTM member is silently renormalized out of the backtest.
- **RL policy members (lstm_ppo / xlstm_ppo / xlstm_grpo)** — pad to `len(X)`,
  so they are kept, but see the performance note below.

**Practical consequence:** for a forecast-only backtest, training with
`--models arima,prophet,xgboost` produces the same backtest blend as
`arima,prophet,lstm,xgboost` (the LSTM is dropped anyway) **and avoids the
per-bar JAX cost below**. Use the 3-member set unless/until the LSTM length
reconciliation lands.

## Performance: per-bar prediction cost

`run_segment` calls `model.predict` once per bar over the WFO test range. Costs
to be aware of on a multi-year daily backtest (~1,600 bars/symbol):

- **JAX members re-trace/compile per call.** A windowed LSTM/RL member triggers
  a fresh JIT trace on most bars (input shape/dtype variation defeats the
  compile cache), which dominates wall-clock — a 4-symbol forecast+LSTM
  backtest can take ~80 min, almost all of it JAX retracing, *for a member
  that's dropped from the blend anyway*. A forecast-only (no-JAX) backtest of
  the same universe runs in ~15-20 min.
- **statsmodels emits a per-bar `ValueWarning`** (no frequency on the index);
  `PYTHONWARNINGS=ignore` quiets Python warnings but statsmodels routes some
  through logging — expect a chatty stderr. Redirect stdout/stderr to a file
  you don't mind growing, and don't run the backtest with `--verbose`.

## Reproducing a run from scratch

```bash
# 1. key (gitignored)
echo "TWELVEDATA_API_KEY=..." > .env

# 2. forecast-only training (no JAX; fast)
.venv/bin/python3 -m scripts.training \
  --symbols AAPL,MSFT,GOOG,AMZN \
  --start_date 2020-01-01 --end_date <today> \
  --horizon 5 --n_splits 5 --models arima,prophet,xgboost

# 3. backtest the run (basic tier -> --no_dividends)
PYTHONWARNINGS=ignore .venv/bin/python3 -m scripts.backtest \
  --training_run runs/<run_name> --no_dividends
```

Artifacts: `runs/<run_name>/{symbol}/fold_*/ensemble_model/` (per-fold models +
`split_idx.npz`), backtest report under `results/wfo_backtest_<ts>/`. Both
`runs/`, `results/`, and `logs/` are gitignored.

## The synthetic-vs-real lesson

Every verification before the first real-data run used synthetic GBM, which
exercised code paths that masked three real bugs (vendor interval string,
ensemble member persistence, backtest predict-frame column set). When changing
the data/model/backtest plumbing, prefer a real (or at least
schema-faithful) fixture over GBM — the regression suite
`tests/test_real_pipeline_regressions.py` now guards these specific paths.
