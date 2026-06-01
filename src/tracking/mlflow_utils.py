"""Thin MLflow wrappers for the WFO outer loop.

Phase 2.2 deliverable: per-fold metric + parameter + artifact logging on
top of MLflow's file-store backend. The tracking URI defaults to
``file://{PROJECT_DIR}/mlruns`` via ``src.config.MLFLOW_TRACKING_URI`` and
can be overridden by setting the ``MLFLOW_TRACKING_URI`` env var (e.g. to
a remote server). Reading these helpers is the cheap path; importing
``mlflow`` directly in scripts is also fine — these only exist to keep the
URI + experiment + non-finite filtering in one place.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import mlflow

from src.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def init_mlflow(experiment_name: str) -> str:
    """Set tracking URI + experiment and return the experiment id.

    Idempotent: multiple calls with the same experiment name resolve to the
    same id and re-setting the URI is a no-op. Callers should ``with
    mlflow.start_run(experiment_id=…)`` (or rely on the default since
    ``set_experiment`` makes it implicit).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow tracking URI={MLFLOW_TRACKING_URI} experiment={experiment_name} "
        f"id={exp.experiment_id}"
    )
    return exp.experiment_id


def log_metrics_safe(
    metrics: Mapping[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log scalar metrics, silently skipping non-finite + non-numeric values.

    mlflow.log_metric raises on NaN/Inf and refuses non-numeric values. The
    ensemble.evaluate dict sometimes contains NaN (e.g. when a member fails
    eval) so the WFO loop needs a tolerant logger to avoid losing the whole
    fold's metrics over one bad scalar.
    """
    for key, value in metrics.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        name = f"{prefix}{key}" if prefix else key
        mlflow.log_metric(name, v, step=step)


def log_params_safe(params: Mapping[str, Any]) -> None:
    """Log parameters as strings, skipping None and complex objects.

    MLflow stores params as strings under the hood with a 6000-char cap.
    Anything that can't be cleanly stringified (large arrays, dataframes)
    should be logged as an artifact instead.
    """
    for key, value in params.items():
        if value is None:
            continue
        try:
            text = str(value)
        except Exception:
            continue
        if len(text) > 6000:
            text = text[:6000]
        mlflow.log_param(key, text)


def log_artifact_dir(path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
    """Log every file under ``path`` as run artifacts.

    Thin wrapper over ``mlflow.log_artifacts`` so callers don't have to
    convert ``Path`` to ``str``. ``artifact_path`` is the subdirectory
    within the run's artifact store (None = root).
    """
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)
