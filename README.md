Project Overview
This project implements a sophisticated trading system that combines classical time series forecasting methods with reinforcement learning to make trading decisions. The core innovation is an ensemble approach that leverages multiple forecasting models including ARIMA, Prophet, LSTM, XGBoost, and three reinforcement learning agents: LSTM-PPO, xLSTM-PPO, and xLSTM-GRPO.

> **Status note (2026-05):** an active remediation effort is underway against `/home/boom90lb/.claude/plans/how-can-we-best-proud-minsky.md`. Phase 1.3 (config cleanup) is complete; the RL training loops, ensemble unit unification, and purged walk-forward CV are pending. Treat backtest numbers as untrustworthy until Phase 1.1–2.1 land.
Core Components

Multi-Model Ensemble - Combines predictions from multiple forecasting models with weighted averaging
LSTM-PPO Model - Reinforcement learning model that optimizes trading decisions
Feature Engineering - Technical indicators and lagged features
Sentiment Analysis - News sentiment integration via Polygon.io API
Backtesting Framework - Comprehensive testing with performance metrics
Real-time Prediction - Making predictions using Twelvedata API

Project Structure
├── src/
│   ├── config.py             # Configuration settings
│   ├── data_loader.py        # Data fetching from Twelvedata
│   ├── features.py           # Feature engineering
│   ├── sentiment_analysis.py # News sentiment analysis using Polygon.io
│   ├── trading.py            # Trading strategy and backtesting
│   ├── models/
│   │   ├── base.py           # Base model class
│   │   ├── arima.py          # ARIMA model
│   │   ├── prophet.py        # Facebook Prophet model
│   │   ├── lstm.py           # LSTM neural network model
│   │   ├── xgboost_model.py  # XGBoost model
│   │   ├── lstm_ppo.py       # LSTM-PPO reinforcement learning model
│   │   ├── xlstm_ppo.py      # xLSTM-PPO reinforcement learning model
│   │   ├── xlstm_grpo.py     # xLSTM-GRPO preference-ranked RL model
│   │   └── ensemble.py       # Ensemble model combining others
├── scripts/
│   ├── training.py           # Script to train models
│   ├── prediction.py         # Script to make predictions
│   └── backtest.py           # Script to backtest the model
Setup Instructions

Environment Setup:
bash# Clone the repository
git clone https://github.com/boom90lb/trading-ensemble.git
cd trading-ensemble

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install using uv for faster dependency resolution
pip install uv
uv pip install -e .

API Configuration:

Create a .env file with your API keys:

TWELVEDATA_API_KEY=your_twelvedata_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here


Usage Guide
Training the Models
bash# Basic training
python -m scripts.train --symbols AAPL,MSFT,GOOG --timeframe 1d --start_date 2020-01-01

# With sentiment analysis
python -m scripts.train --symbols AAPL --timeframe 1d --start_date 2020-01-01 --use_sentiment

# Specific models with custom RL training steps
python -m scripts.train --symbols AAPL --models arima,lstm,lstm_ppo --rl_timesteps 200000
Making Predictions
bash# Basic prediction
python -m scripts.predict --symbols AAPL --timeframe 1d --horizon 5

# With sentiment analysis and visualization
python -m scripts.predict --symbols AAPL --use_sentiment --plot
Backtesting
bash# Simple backtest
python -m scripts.backtest --symbols AAPL --start_date 2022-01-01 --end_date 2023-01-01

# With LSTM-PPO model and sentiment analysis
python -m scripts.backtest --symbols AAPL --start_date 2022-01-01 --use_sentiment --use_ppo

# Compare ensemble vs LSTM-PPO performance
python -m scripts.backtest --symbols all --start_date 2022-01-01 --use_ppo
Key Implementation Details
LSTM-PPO Architecture
The LSTM-PPO model in src/models/lstm_ppo.py combines:

LSTM Feature Extractor: Processes sequential market data
PPO Algorithm: Optimizes trading decisions
Custom Gym Environment: Simulates trading with realistic constraints

The trading environment:

Takes actions in range [-1.0, 1.0] for position sizing
Calculates rewards based on PnL and transaction costs
Uses a sliding window of features as observation space

Ensemble Model
The ensemble in src/models/ensemble.py combines predictions through:

Static or dynamic weighting strategies
Adaptive contribution based on recent performance
Individual model optimization

Trading Strategy
The trading strategy in src/trading.py:

Calculates signals based on model predictions and sentiment
Executes trades with configurable position sizing
Implements stop-loss and take-profit mechanisms
Calculates comprehensive performance metrics
Generates visualizations and reports

Performance Metrics
The backtesting framework calculates:

Total Return
Annualized Return
Sharpe Ratio
Maximum Drawdown
Win Rate
Profit Factor

Dependencies

Python 3.12+
Data APIs: Twelvedata, Polygon.io
ML: scikit-learn, Flax, JAX, PyTorch
RL: Gymnasium, Stable-Baselines3
Forecasting: statsmodels, Prophet, XGBoost

Configuration

`src/config.py` is the single source of truth for ensemble weights, training hyperparameters, and trading parameters.

- `DEFAULT_MODEL_WEIGHTS` registers every ensemble member, including `lstm_ppo`, `xlstm_ppo`, and `xlstm_grpo`. Scripts must read weights from this dict rather than hardcoding per-model overrides.
- `TrainingConfig` and `TradingConfig` validate their fields in `__post_init__` (e.g. `0 < train_test_split < 1`, `0 < position_size <= 1`, `0 <= commission < 0.05`). Invalid CLI arguments fail fast at config construction.
- `JAX_CONFIG["device"]` defaults to `"auto"`; resolve a concrete backend with `resolve_jax_device()` (returns `gpu` / `tpu` / `cpu` based on `jax.devices()`).
- `MLFLOW_TRACKING_URI` falls back to a local `file://.../mlruns` store; override with the environment variable to point at a remote tracking server.

Data Quality

Corporate actions (Phase 4 §4.1). Prices are fetched `adjust=splits` — split-adjusted but **not** dividend-adjusted — so the close series is a faithful tradeable price. Dividends are handled as **explicit cash** in the backtest (`DataLoader.fetch_dividends` → `TradingStrategy.apply_dividends`): a long held over an ex-date takes a mark-to-market markdown that the per-share credit offsets (a short is debited), making the position total-return correct without back-adjusting prices (which would inject a price-level look-ahead and churn the cache). Trade-off: ex-dividend gaps remain in the return/volatility *features* — the accepted cost of this choice. Use `--no_dividends` on the backtest to disable the credit (and the `/dividends` API calls).

Cache integrity (Phase 4 §4.4). Cached bars are keyed by the **requested date range** (`{symbol}_{interval}_{start}_{end}.parquet`); a fetch reuses a cached file only when its range *contains* the request, so a narrow cache can no longer silently satisfy a wider query with a too-short slice. Coverage is decided on the filename range (not `index.min()/max()`), so short-history symbols don't force endless refetches.

Universe and survivorship (Phase 4 §4.3). The investable set is defined at **training** time: pass `--universe data/universe/{date}.txt` (one symbol per line, `#` comments) and optionally `--universe_asof YYYY-MM-DD` to drop names with no data at/before that date (a forward-looking-inclusion guard). This is **best-effort** point-in-time construction on the *included* names only — it does not recover delisted or acquired tickers. True survivorship-bias-free universes require a delisting database (CRSP, Norgate); integrating one is **out of scope** for this project, and backtest results should be read with that residual survivorship bias in mind.

License
MIT License, Copyright (c) 2026 Brendon Reperttang