# Time Series Ensemble Trading Model with LSTM-PPO

A sophisticated trading system that combines classical time series forecasting methods with reinforcement learning to make trading decisions. This project implements an ensemble of forecasting models including ARIMA, Prophet, LSTM, XGBoost, and a custom LSTM-PPO reinforcement learning model for optimized trading decision making.

## Features

- **Multi-model ensemble** approach combining classical time series models and machine learning
- **LSTM-PPO reinforcement learning** for optimized trading decisions
- **News sentiment analysis** integration via Polygon.io API
- **Comprehensive backtesting** framework
- **Feature engineering** pipeline with technical indicators
- **Real-time prediction** capabilities using Twelvedata API

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_loader.py        # Data fetching from Twelvedata
│   ├── features.py           # Feature engineering
│   ├── sentiment_analysis.py # News sentiment analysis using Polygon.io
│   ├── trading.py            # Trading strategy and backtesting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # Base model class
│   │   ├── arima.py          # ARIMA model
│   │   ├── prophet.py        # Facebook Prophet model
│   │   ├── lstm.py           # LSTM neural network model
│   │   ├── xgboost_model.py  # XGBoost model
│   │   ├── lstm_ppo.py       # LSTM-PPO reinforcement learning model
│   │   └── ensemble.py       # Ensemble model combining others
├── scripts/
│   ├── training.py              # Script to train models
│   ├── prediction.py            # Script to make predictions
│   └── backtest.py           # Script to backtest the model
├── data/                     # Directory for stored data
├── models/                   # Directory for saved models
├── results/                  # Directory for results and plots
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Installation

This project requires Python 3.12 or higher.

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-ensemble.git
cd trading-ensemble

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install using uv for faster dependency resolution
pip install uv
uv pip install -e .

# Install the latest versions of CUDA and cuDNN
[NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html)
```

## Configuration

Create a `.env` file in the project root with your API keys:

```
TWELVEDATA_API_KEY=your_twelvedata_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
```

## Usage

### Training the Models

```bash
# Train the ensemble model with default settings
python -m scripts.train --symbols AAPL,MSFT,GOOG --timeframe 1day --start_date 2020-01-01

# Train with sentiment analysis
python -m scripts.train --symbols AAPL --timeframe 1day --start_date 2020-01-01 --use_sentiment

# Train with specified models and RL timesteps
python -m scripts.train --symbols AAPL --models arima,lstm,lstm_ppo --rl_timesteps 200000
```

### Making Predictions

```bash
# Make predictions for a symbol
python -m scripts.predict --symbols AAPL --timeframe 1day --horizon 5

# Make predictions with sentiment analysis and plot results
python -m scripts.predict --symbols AAPL --use_sentiment --plot
```

### Backtesting

```bash
# Run a backtest on historical data
python -m scripts.backtest --symbols AAPL --start_date 2022-01-01 --end_date 2023-01-01

# Backtest with LSTM-PPO model and sentiment analysis
python -m scripts.backtest --symbols AAPL --start_date 2022-01-01 --use_sentiment --use_ppo

# Compare ensemble vs LSTM-PPO performance
python -m scripts.backtest --symbols all --start_date 2022-01-01 --use_ppo
```

## Model Architecture

### Ensemble Model

The ensemble model combines predictions from multiple forecasting models:

1. **ARIMA**: Classical statistical model for time series forecasting
2. **Prophet**: Facebook's time series forecasting model
3. **LSTM**: Long Short-Term Memory neural network
4. **XGBoost**: Gradient boosting for time series regression
5. **LSTM-PPO**: Reinforcement learning model for trading decisions

### LSTM-PPO Model

The LSTM-PPO model combines:

1. **LSTM Feature Extractor**: Processes sequential market data
2. **PPO Algorithm**: Proximal Policy Optimization for trading decisions
3. **Custom Gym Environment**: Trading environment with realistic constraints

## Sentiment Analysis

The system integrates news sentiment analysis from Polygon.io:

- **Sentiment Scores**: Numeric representations of news sentiment
- **Sentiment Momentum**: Changes in sentiment over time
- **News Volume**: Tracking news activity levels

## Performance Metrics

The backtesting framework calculates:

- **Total Return**: Overall performance
- **Annualized Return**: Yearly performance
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses

## License

MIT

## Acknowledgements

- [Twelvedata](https://twelvedata.com/) for market data
- [Polygon.io](https://polygon.io/) for news sentiment data
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL implementation
- [Gymnasium](https://gymnasium.farama.org/) for the RL environment framework
