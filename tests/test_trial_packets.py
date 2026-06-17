"""Tests for canonical research-trial claim packets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.validation.trials import (
    CLAIM_GROSS_EDGE,
    CLAIM_MECHANICS_CLEAN,
    CLAIM_NET_EDGE,
    CLAIM_NO_CLAIM,
    CLAIM_ROBUST_EDGE,
    REQUIRED_PACKET_KEYS,
    build_research_claim_packet,
    classify_claim_tier,
    emit_research_claim_packet,
    stable_config_hash,
    summary_claim_fields,
    validate_claim_packet_dir,
    write_research_claim_packet,
)


def test_stable_config_hash_is_key_order_invariant() -> None:
    left = {"b": 2, "a": {"y": 4, "x": [1, 2, 3]}}
    right = {"a": {"x": [1, 2, 3], "y": 4}, "b": 2}
    assert stable_config_hash(left) == stable_config_hash(right)


def test_claim_tier_requires_enough_observations() -> None:
    assert classify_claim_tier({"n_obs": 2, "periodic_sharpe": 1.0}, min_obs=20) == CLAIM_NO_CLAIM


def test_claim_tier_distinguishes_gross_and_net_edge() -> None:
    metrics = {
        "n_obs": 40,
        "gross_total_return": 0.05,
        "gross_periodic_sharpe": 0.10,
        "total_return": -0.01,
        "periodic_sharpe": -0.02,
    }
    assert classify_claim_tier(metrics) == CLAIM_GROSS_EDGE

    metrics.update({"total_return": 0.02, "periodic_sharpe": 0.05})
    assert classify_claim_tier(metrics) == CLAIM_NET_EDGE

    metrics["dsr"] = 0.97
    assert classify_claim_tier(metrics) == CLAIM_ROBUST_EDGE


def test_build_packet_computes_cost_adjusted_gross_metrics() -> None:
    idx = pd.date_range("2026-01-01", periods=30, freq="B", tz="America/New_York")
    net_values = np.where(np.arange(len(idx)) % 2 == 0, -0.0020, 0.0004)
    net = pd.Series(net_values, index=idx)
    costs = pd.DataFrame({"total": np.full(len(idx), 0.0012), "turnover": 0.2}, index=idx)
    weights = pd.DataFrame({"AAA": 0.5, "BBB": -0.5}, index=idx)

    packet = build_research_claim_packet(
        strategy="unit_test_strategy",
        config={"param": 1},
        data={"symbols": ["AAA", "BBB"]},
        returns=net,
        costs=costs,
        target_weights=weights,
        summary={"pair_set_dsr": 0.1},
        code_commit="abc123",
    )

    assert packet.claim_tier == CLAIM_GROSS_EDGE
    assert packet.metrics["total_return"] < 0
    assert packet.metrics["gross_total_return"] > 0
    assert packet.metrics["total_cost"] == costs["total"].sum()
    assert packet.metrics["n_target_symbols"] == 2
    assert packet.code_commit == "abc123"


def test_packet_writer_emits_strict_json(tmp_path) -> None:
    packet = build_research_claim_packet(
        strategy="flat",
        config={"param": float("nan")},
        data={"start": pd.Timestamp("2026-01-01", tz="UTC")},
        returns=[0.0] * 30,
        summary={"dsr": float("nan")},
        artifacts={"returns": "returns.csv"},
    )
    assert packet.claim_tier == CLAIM_MECHANICS_CLEAN

    out = tmp_path / "claim_packet.json"
    write_research_claim_packet(packet, out)
    loaded = json.loads(out.read_text())

    assert loaded["schema_version"] == 1
    assert loaded["config"]["param"] is None
    assert loaded["metrics"]["dsr"] is None
    assert loaded["artifacts"]["returns"] == "returns.csv"


def test_summary_claim_fields_match_packet() -> None:
    packet = build_research_claim_packet(
        strategy="unit", config={"a": 1}, data={}, returns=[0.0] * 30,
    )
    fields = summary_claim_fields(packet)
    assert fields["claim_tier"] == packet.claim_tier
    assert fields["config_hash"] == packet.config_hash
    assert fields["claim_packet_schema_version"] == packet.schema_version
    assert fields["claim_packet"] == "claim_packet.json"


def test_emit_writes_packet_and_returns_it(tmp_path) -> None:
    packet = emit_research_claim_packet(
        tmp_path,
        strategy="unit", config={}, data={}, returns=[0.0] * 30,
        artifacts={"claim_packet": "claim_packet.json"},
    )
    written = json.loads((tmp_path / "claim_packet.json").read_text())
    assert written["config_hash"] == packet.config_hash
    assert written["strategy"] == "unit"


def test_validate_accepts_well_formed_dir(tmp_path) -> None:
    (tmp_path / "returns.csv").write_text("idx,r\n0,0.0\n")
    packet = emit_research_claim_packet(
        tmp_path,
        strategy="unit", config={"a": 1}, data={"symbols": ["AAA"]},
        returns=[0.0] * 30,
        artifacts={"returns": "returns.csv", "claim_packet": "claim_packet.json"},
    )
    loaded = validate_claim_packet_dir(tmp_path)
    assert loaded["strategy"] == "unit"
    assert loaded["claim_tier"] == packet.claim_tier
    # Every required key is present in the validated packet.
    assert set(REQUIRED_PACKET_KEYS).issubset(loaded)


def test_validate_rejects_missing_packet(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_claim_packet_dir(tmp_path)


def test_validate_rejects_missing_artifact(tmp_path) -> None:
    # returns.csv is named in the manifest but never written to disk.
    emit_research_claim_packet(
        tmp_path,
        strategy="unit", config={}, data={}, returns=[0.0] * 30,
        artifacts={"returns": "returns.csv", "claim_packet": "claim_packet.json"},
    )
    with pytest.raises(ValueError, match="missing artifacts"):
        validate_claim_packet_dir(tmp_path)


def test_validate_rejects_unknown_tier(tmp_path) -> None:
    emit_research_claim_packet(
        tmp_path, strategy="unit", config={}, data={}, returns=[0.0] * 30,
        artifacts={"claim_packet": "claim_packet.json"},
    )
    path = tmp_path / "claim_packet.json"
    obj = json.loads(path.read_text())
    obj["claim_tier"] = "totally_made_up"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="unknown claim_tier"):
        validate_claim_packet_dir(tmp_path)


def test_validate_rejects_non_finite_json(tmp_path) -> None:
    emit_research_claim_packet(
        tmp_path, strategy="unit", config={}, data={}, returns=[0.0] * 30,
        artifacts={"claim_packet": "claim_packet.json"},
    )
    path = tmp_path / "claim_packet.json"
    obj = json.loads(path.read_text())
    obj["metrics"]["periodic_sharpe"] = float("nan")
    # Default json.dumps emits a bare ``NaN`` token, which strict parsing rejects.
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        validate_claim_packet_dir(tmp_path)
