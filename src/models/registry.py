"""Model-type registry.

Splits ensemble members by what their `predict` actually returns so the
ensemble can stop averaging dimensionally incoherent outputs (B8). Forecast
models emit expected h-bar returns; policy models emit position ∈ [-1, 1].
"""

from typing import Literal

FORECAST_MODELS: frozenset[str] = frozenset({"arima", "prophet", "xgboost"})
POLICY_MODELS: frozenset[str] = frozenset({"lstm_ppo", "xlstm_ppo", "xlstm_grpo"})

ModelKind = Literal["forecast", "policy"]


def model_kind(name: str) -> ModelKind:
    if name in FORECAST_MODELS:
        return "forecast"
    if name in POLICY_MODELS:
        return "policy"
    raise ValueError(
        f"Unknown model '{name}'. Register it in FORECAST_MODELS or POLICY_MODELS."
    )


def is_forecast(name: str) -> bool:
    return name in FORECAST_MODELS


def is_policy(name: str) -> bool:
    return name in POLICY_MODELS
