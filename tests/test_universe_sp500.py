"""Survivorship-bias-free S&P 500 universe reconstruction (Koker method).

The headline invariant under test: names removed from the index still appear in
the ever-member union, and if their prices are unavailable they are *counted* in
the coverage skip-list -- survivorship is measured, never silently dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.universe_sp500 import (
    HISTORY_FLOOR,
    CoverageReport,
    build_membership_mask,
    compute_coverage,
    ever_members,
    extract_tables,
    members_active_between,
    parse_changes_table,
    parse_constituents_table,
    reconstruct_membership,
    write_universe_file,
)

END = pd.Timestamp("2026-01-01")


def _changes(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "added", "removed"])


def _interval(df: pd.DataFrame, ticker: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    g = df[df["ticker"] == ticker].sort_values("start")
    return [(pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in g.itertuples(index=False)]


def test_member_never_in_changes_spans_full_history() -> None:
    m = reconstruct_membership(["A"], _changes([]), end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]


def test_added_member_and_displaced_member_intervals() -> None:
    # B replaced X on 2018-06-01: X member until then, B member after.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]
    assert _interval(m, "B") == [(pd.Timestamp("2018-06-01"), END)]
    assert _interval(m, "X") == [(HISTORY_FLOOR, pd.Timestamp("2018-06-01"))]


def test_readdition_produces_two_disjoint_intervals() -> None:
    # A: added 2010, removed 2015, added back 2020; current member.
    changes = _changes(
        [("2010-01-04", "A", ""), ("2015-03-02", "", "A"), ("2020-09-01", "A", "")]
    )
    m = reconstruct_membership(["A"], changes, end_date=END)
    assert _interval(m, "A") == [
        (pd.Timestamp("2010-01-04"), pd.Timestamp("2015-03-02")),
        (pd.Timestamp("2020-09-01"), END),
    ]


def test_ever_members_includes_removed_names_survivorship() -> None:
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    # X left the index in 2018 but MUST remain in the universe.
    assert ever_members(m) == ["A", "B", "X"]


def test_members_active_between_scopes_to_window() -> None:
    # X left 2018, B joined 2018, A always. A 2020+ window excludes the
    # pre-window casualty X but keeps it survivorship-free *within* the window.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    assert members_active_between(m, "2020-01-01", "2026-01-01") == ["A", "B"]
    # A window that straddles the 2018 change still includes X (member then).
    assert "X" in members_active_between(m, "2017-01-01", "2018-12-31")


def test_inconsistent_events_are_skipped_not_fatal() -> None:
    # Two genuinely inconsistent events that must be skipped (not crash, not
    # spawn phantom intervals): a duplicate add of an already-active member (C
    # at 2016), and a second removal of an already-removed name (Q at 2019).
    changes = _changes(
        [
            ("2015-01-05", "C", ""),
            ("2016-06-01", "C", ""),  # C already active -> duplicate add skipped
            ("2018-03-01", "", "Q"),
            ("2019-04-01", "", "Q"),  # Q already removed -> double removal skipped
        ]
    )
    m = reconstruct_membership(["A", "C"], changes, end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]
    assert _interval(m, "C") == [(pd.Timestamp("2015-01-05"), END)]  # single, not doubled
    assert _interval(m, "Q") == [(HISTORY_FLOOR, pd.Timestamp("2018-03-01"))]  # closed once


def test_future_dated_addition_excluded_as_of_horizon() -> None:
    # A change announced for after the horizon (END = 2026): the incoming name is
    # not yet a member (no inverted interval); the outgoing name still is.
    changes = _changes([("2030-01-01", "NEW", "OLD")])
    m = reconstruct_membership(["A"], changes, end_date=END)
    assert "NEW" not in set(m["ticker"])
    assert _interval(m, "OLD") == [(HISTORY_FLOOR, pd.Timestamp("2030-01-01"))]
    assert "NEW" not in set(ever_members(m))


def test_reconstruct_accepts_tz_aware_end_date() -> None:
    # Regression: a tz-aware horizon (pd.Timestamp.utcnow()) must not clash with
    # tz-naive Wikipedia change dates.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=pd.Timestamp("2026-01-01", tz="UTC"))
    assert set(ever_members(m)) == {"A", "B", "X"}
    assert (m["start"] <= m["end"]).all()


def test_membership_mask_is_time_varying_and_tz_safe() -> None:
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    idx = pd.date_range("2018-01-01", "2018-12-31", freq="B", tz="America/New_York")
    mask = build_membership_mask(m, idx, ["A", "B", "X"])

    early = pd.Timestamp("2018-03-01", tz="America/New_York")
    late = pd.Timestamp("2018-09-03", tz="America/New_York")
    assert mask.loc[early, "A"] and mask.loc[early, "X"] and not mask.loc[early, "B"]
    assert mask.loc[late, "A"] and mask.loc[late, "B"] and not mask.loc[late, "X"]
    # A name absent from the universe altogether is never a member.
    mask2 = build_membership_mask(m, idx, ["A", "ZZZZ"])
    assert not mask2["ZZZZ"].any()


def test_coverage_counts_unavailable_delisted_names() -> None:
    ever = ["A", "B", "X"]
    # X is delisted; the price vendor returned no bars for it.
    report = compute_coverage(ever, available=["A", "B"], asof="2026-01-01")
    assert isinstance(report, CoverageReport)
    assert report.skipped == ["X"]
    assert report.resolved == ["A", "B"]
    assert report.n_ever_members == 3 and report.n_resolved == 2
    assert report.coverage_fraction == pytest.approx(2 / 3)


def test_coverage_applies_rename_table() -> None:
    report = compute_coverage(
        ["BRK.B"], available=["BRK/B"], asof="2026-01-01", rename_table={"BRK.B": "BRK/B"}
    )
    assert report.resolved == ["BRK.B"] and report.skipped == []


CONSTITUENTS_HTML = """
<table class="wikitable" id="constituents">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
  <tr><td>AOS</td><td>A. O. Smith</td><td>Industrials</td></tr>
  <tr><td>NVDA</td><td>Nvidia<sup>[1]</sup></td><td>Information Technology</td></tr>
</table>
<table class="wikitable" id="changes">
  <tr><th>Date</th><th>Added</th><th>Removed</th><th>Reason</th></tr>
  <tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
  <tr><td>June 20, 2018</td><td>B</td><td>Beta Co</td><td>X</td><td>Xenon Inc</td><td>M&amp;A</td></tr>
  <tr><td>March 2, 2015</td><td></td><td></td><td>A</td><td>Alpha</td><td>Bankruptcy[2]</td></tr>
</table>
"""


def test_extract_and_parse_wikipedia_tables() -> None:
    tables = extract_tables(CONSTITUENTS_HTML)
    assert parse_constituents_table(tables) == ["MMM", "AOS", "NVDA"]
    changes = parse_changes_table(tables)
    # Two parseable change rows; sub-header ('Ticker'/'Security') row dropped.
    assert list(changes.columns) == ["date", "added", "removed"]
    assert len(changes) == 2
    first = changes.iloc[0]
    assert first["date"] == pd.Timestamp("2018-06-20")
    assert first["added"] == "B" and first["removed"] == "X"


def test_end_to_end_parse_then_reconstruct() -> None:
    tables = extract_tables(CONSTITUENTS_HTML)
    current = parse_constituents_table(tables)
    changes = parse_changes_table(tables)
    m = reconstruct_membership(current, changes, end_date=END)
    union = set(ever_members(m))
    # Current names plus both removed names survive into the universe.
    assert {"MMM", "AOS", "NVDA", "X", "A"} <= union


def test_write_universe_file_matches_load_universe_contract(tmp_path) -> None:
    path = tmp_path / "sp500_pit_2026-01-01.txt"
    write_universe_file(["B", "A", "x", "A"], path, asof="2026-01-01")
    # Mirror scripts.training.load_universe: strip '#' comments, dedupe in order.
    loaded: list[str] = []
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token and token not in loaded:
            loaded.append(token)
    assert loaded == ["A", "B", "X"]
