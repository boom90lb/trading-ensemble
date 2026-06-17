"""Canonical research-trial packets and claim-tier classification.

The project has several strategy surfaces that already write their own JSON/CSV
artifacts. This module gives them one shared, strict packet shape for the
methodology layer: what was run, which data convention it used, which artifacts
prove it, and what claim tier the observed result supports.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.validation.metrics import periodic_sharpe

CLAIM_NO_CLAIM = "no_claim"
CLAIM_MECHANICS_CLEAN = "mechanics_clean"
CLAIM_GROSS_EDGE = "gross_edge"
CLAIM_NET_EDGE = "net_edge"
CLAIM_ROBUST_EDGE = "robust_edge"

CLAIM_TIERS = (
    CLAIM_NO_CLAIM,
    CLAIM_MECHANICS_CLEAN,
    CLAIM_GROSS_EDGE,
    CLAIM_NET_EDGE,
    CLAIM_ROBUST_EDGE,
)

TRIAL_PACKET_SCHEMA_VERSION = 1
DEFAULT_ROBUST_DSR_THRESHOLD = 0.95
DEFAULT_MIN_OBS = 20


@dataclass(frozen=True)
class ResearchClaimPacket:
    """Strict JSON-compatible summary of one research trial."""

    schema_version: int
    created_at_utc: str
    strategy: str
    claim_tier: str
    config_hash: str
    code_commit: str | None
    data: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    claim_rules: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible packet dict."""
        return _to_jsonable(asdict(self))


def _to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if dataclass_is_instance(value):
        return _to_jsonable(asdict(value))
    return str(value)


def dataclass_is_instance(value: Any) -> bool:
    """True for dataclass instances, false for dataclass classes."""
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """Canonical JSON used for research config identity."""
    return json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_config_hash(payload: Mapping[str, Any], length: int = 12) -> str:
    """Short SHA-256 digest for a canonicalized trial config payload."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:length]


def current_git_commit(cwd: Path | str | None = None) -> str | None:
    """Return the current git commit, or None when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _finite_returns(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    series = pd.Series(returns, dtype=float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _total_return(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    value = float((1.0 + returns).prod() - 1.0)
    return value if math.isfinite(value) else None


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_metric(summary: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in summary:
            value = _optional_float(summary[key])
            if value is not None:
                return value
    return None


def _gross_returns(net_returns: pd.Series, costs: pd.DataFrame | None) -> pd.Series | None:
    if costs is None or "total" not in costs.columns:
        return None
    total_cost = pd.to_numeric(costs["total"], errors="coerce")
    aligned_cost = total_cost.reindex(net_returns.index).fillna(0.0)
    return net_returns.add(aligned_cost, fill_value=0.0)


def _cost_column_mean(costs: pd.DataFrame | None, column: str) -> float | None:
    if costs is None or column not in costs.columns:
        return None
    return _optional_float(pd.to_numeric(costs[column], errors="coerce").mean())


def _cost_column_sum(costs: pd.DataFrame | None, column: str) -> float | None:
    if costs is None or column not in costs.columns:
        return None
    return _optional_float(pd.to_numeric(costs[column], errors="coerce").sum())


def _target_weight_metrics(target_weights: pd.DataFrame | None) -> dict[str, Any]:
    if target_weights is None or target_weights.empty:
        return {}
    weights = target_weights.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    return {
        "n_target_rows": int(len(weights)),
        "n_target_symbols": int(len(weights.columns)),
        "avg_target_gross": _optional_float(gross.mean()),
        "max_target_gross": _optional_float(gross.max()),
    }


def compute_trial_metrics(
    returns: pd.Series | np.ndarray | list[float],
    costs: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute canonical gross/net/cost/trial metrics for one run."""
    summary = summary or {}
    net = _finite_returns(returns)
    net_periodic = periodic_sharpe(net)
    gross = _gross_returns(net, costs)
    gross_periodic = periodic_sharpe(gross) if gross is not None else float("nan")
    total_cost = _cost_column_sum(costs, "total")
    avg_turnover = _cost_column_mean(costs, "turnover")
    avg_gross = _cost_column_mean(costs, "gross")
    avg_net = _cost_column_mean(costs, "net")

    metrics: dict[str, Any] = {
        "n_obs": int(len(net)),
        "total_return": _total_return(net),
        "periodic_sharpe": _optional_float(net_periodic),
        "annualized_sharpe": _optional_float(net_periodic * math.sqrt(252.0)),
        "gross_total_return": _total_return(gross) if gross is not None else None,
        "gross_periodic_sharpe": _optional_float(gross_periodic),
        "gross_annualized_sharpe": _optional_float(gross_periodic * math.sqrt(252.0)),
        "total_cost": total_cost if total_cost is not None else _first_metric(summary, ("total_cost",)),
        "avg_turnover": (
            avg_turnover if avg_turnover is not None else _first_metric(summary, ("avg_turnover", "turnover"))
        ),
        "avg_gross": avg_gross if avg_gross is not None else _first_metric(summary, ("avg_gross",)),
        "avg_net": avg_net if avg_net is not None else _first_metric(summary, ("avg_net",)),
        "dsr": _first_metric(summary, ("dsr", "pair_set_dsr", "residual_set_dsr")),
        "trial_count": _first_metric(
            summary,
            ("trial_count", "n_trials", "ledger_trials", "pair_trial_count", "finite_pair_trial_sharpes"),
        ),
    }
    metrics.update(_target_weight_metrics(target_weights))
    return _to_jsonable(metrics)


def classify_claim_tier(
    metrics: Mapping[str, Any],
    *,
    robust_dsr_threshold: float = DEFAULT_ROBUST_DSR_THRESHOLD,
    min_obs: int = DEFAULT_MIN_OBS,
) -> str:
    """Classify the strongest claim supported by canonical trial metrics."""
    if min_obs < 1:
        raise ValueError(f"min_obs must be >= 1, got {min_obs}")
    if not 0.0 <= robust_dsr_threshold <= 1.0:
        raise ValueError(f"robust_dsr_threshold must be in [0, 1], got {robust_dsr_threshold}")

    n_obs = int(metrics.get("n_obs") or 0)
    if n_obs < min_obs:
        return CLAIM_NO_CLAIM

    net_return = _optional_float(metrics.get("total_return"))
    net_sharpe = _optional_float(metrics.get("periodic_sharpe"))
    gross_return = _optional_float(metrics.get("gross_total_return"))
    gross_sharpe = _optional_float(metrics.get("gross_periodic_sharpe"))
    dsr = _optional_float(metrics.get("dsr"))

    tier = CLAIM_MECHANICS_CLEAN
    if gross_return is not None and gross_sharpe is not None and gross_return > 0 and gross_sharpe > 0:
        tier = CLAIM_GROSS_EDGE
    if net_return is not None and net_sharpe is not None and net_return > 0 and net_sharpe > 0:
        tier = CLAIM_NET_EDGE
    if tier == CLAIM_NET_EDGE and dsr is not None and dsr >= robust_dsr_threshold:
        tier = CLAIM_ROBUST_EDGE
    return tier


def build_research_claim_packet(
    *,
    strategy: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    returns: pd.Series | np.ndarray | list[float],
    costs: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    code_commit: str | None = None,
    created_at: datetime | None = None,
    robust_dsr_threshold: float = DEFAULT_ROBUST_DSR_THRESHOLD,
    min_obs: int = DEFAULT_MIN_OBS,
) -> ResearchClaimPacket:
    """Build one canonical packet for a strategy trial."""
    if not strategy:
        raise ValueError("strategy must be non-empty")
    metrics = compute_trial_metrics(
        returns=returns,
        costs=costs,
        target_weights=target_weights,
        summary=summary,
    )
    tier = classify_claim_tier(
        metrics,
        robust_dsr_threshold=robust_dsr_threshold,
        min_obs=min_obs,
    )
    config_payload = {"strategy": strategy, "config": config, "data": data}
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return ResearchClaimPacket(
        schema_version=TRIAL_PACKET_SCHEMA_VERSION,
        created_at_utc=timestamp.astimezone(timezone.utc).isoformat(),
        strategy=strategy,
        claim_tier=tier,
        config_hash=stable_config_hash(config_payload),
        code_commit=code_commit,
        data=_to_jsonable(data),
        config=_to_jsonable(config),
        metrics=metrics,
        artifacts={str(k): str(v) for k, v in sorted((artifacts or {}).items())},
        claim_rules={
            "claim_tiers": list(CLAIM_TIERS),
            "min_obs": int(min_obs),
            "robust_dsr_threshold": float(robust_dsr_threshold),
            "gross_edge": "gross_total_return > 0 and gross_periodic_sharpe > 0",
            "net_edge": "total_return > 0 and periodic_sharpe > 0",
            "robust_edge": "net_edge and dsr >= robust_dsr_threshold",
        },
    )


def write_research_claim_packet(packet: ResearchClaimPacket, path: Path | str) -> None:
    """Write a strict JSON claim packet."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit_research_claim_packet(
    out_dir: Path | str,
    *,
    filename: str = "claim_packet.json",
    **packet_kwargs: Any,
) -> ResearchClaimPacket:
    """Build a packet and write it as ``out_dir/filename``; return the packet.

    Thin wrapper over :func:`build_research_claim_packet` +
    :func:`write_research_claim_packet` so every strategy surface emits packets
    through one identical path.
    """
    packet = build_research_claim_packet(**packet_kwargs)
    write_research_claim_packet(packet, Path(out_dir) / filename)
    return packet


def summary_claim_fields(
    packet: ResearchClaimPacket,
    *,
    packet_filename: str = "claim_packet.json",
) -> dict[str, Any]:
    """Canonical claim-packet fields to fold into a strategy's summary JSON."""
    return {
        "claim_tier": packet.claim_tier,
        "config_hash": packet.config_hash,
        "claim_packet_schema_version": packet.schema_version,
        "claim_packet": packet_filename,
    }


REQUIRED_PACKET_KEYS: tuple[str, ...] = (
    "schema_version",
    "created_at_utc",
    "strategy",
    "claim_tier",
    "config_hash",
    "data",
    "config",
    "metrics",
    "artifacts",
    "claim_rules",
)


def _reject_non_finite_json(token: str) -> Any:
    """``json.loads`` ``parse_constant`` hook: a strict packet has no NaN/Inf."""
    raise ValueError(f"non-finite JSON token {token!r} is not allowed in a strict claim packet")


def validate_claim_packet_dir(
    out_dir: Path | str,
    *,
    packet_filename: str = "claim_packet.json",
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a strategy output dir against the research-claim contract.

    Asserts the dir holds a strict-JSON packet (no NaN/Inf), with every required
    key, a known ``claim_tier``, the current schema version, and — when
    ``require_artifacts`` — that every file named in the packet's ``artifacts``
    manifest actually exists in ``out_dir``. Returns the parsed packet dict.
    Raises ``FileNotFoundError``/``ValueError`` on the first violation.
    """
    out = Path(out_dir)
    packet_path = out / packet_filename
    if not packet_path.is_file():
        raise FileNotFoundError(f"claim packet not found: {packet_path}")

    packet = json.loads(packet_path.read_text(), parse_constant=_reject_non_finite_json)
    if not isinstance(packet, dict):
        raise ValueError(f"claim packet is not a JSON object: {packet_path}")

    missing_keys = [key for key in REQUIRED_PACKET_KEYS if key not in packet]
    if missing_keys:
        raise ValueError(f"claim packet missing required keys {missing_keys}: {packet_path}")

    if packet["schema_version"] != TRIAL_PACKET_SCHEMA_VERSION:
        raise ValueError(
            f"claim packet schema_version {packet['schema_version']} != "
            f"{TRIAL_PACKET_SCHEMA_VERSION}: {packet_path}"
        )
    if packet["claim_tier"] not in CLAIM_TIERS:
        raise ValueError(f"claim packet has unknown claim_tier {packet['claim_tier']!r}: {packet_path}")

    artifacts = packet["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"claim packet artifacts is not an object: {packet_path}")
    if require_artifacts:
        missing = sorted(
            rel
            for rel in artifacts.values()
            if rel != packet_filename and not (out / rel).is_file()
        )
        if missing:
            raise ValueError(f"claim packet references missing artifacts {missing} under {out}")
    return packet
