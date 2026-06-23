"""Survivorship-bias-free S&P 500 universe reconstruction (Koker method).

Point-in-time index membership is reconstructed from the Wikipedia "List of
S&P 500 companies" page: the current-constituents table plus the "Selected
changes to the list" table (date, added ticker, removed ticker). Walking the
changes backward from the scrape date yields the member set on any historical
date; the union over all dates is the survivorship-bias-free universe -- it
*includes* names since removed by delisting, acquisition, or relegation, which
a current-constituents snapshot would silently drop.

Price coverage is a separate, measured concern. Free vendors do not always
retain prices for delisted tickers, so the builder emits an explicit coverage
ledger (resolved vs skipped) rather than quietly dropping the dead names that
make the universe survivorship-free in the first place. Re-introducing
survivorship through missing delisted prices is only acceptable if it is
*counted* -- see ``compute_coverage`` and ``scripts/build_sp500_universe.py``.

Dependency note: the project env has neither lxml nor bs4, so ``pandas.read_html``
is unavailable. The two tables are extracted with a small stdlib ``html.parser``
subclass, and every reconstruction function operates on plain DataFrames so the
logic is unit-testable without network.

Reference: Teddy Koker, "Creating a Survivorship-Bias-Free S&P 500 Dataset with
Python" (2019). Koker used iShares IVV monthly holdings as the membership proxy;
this module uses the Wikipedia changes table (longer history, single stable
scrape) and keeps the same explicit rename-table / skip-set discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Interval floor for names that were already members before the earliest row in
# the changes table (membership "since before records"). Chosen well before any
# backtest sample so masking treats it as -inf.
HISTORY_FLOOR = pd.Timestamp("1990-01-01")

# Manual, *reviewed* ticker remaps from Wikipedia spelling to the price vendor's
# spelling (Koker's `rename_table`). Seeded empty: entries are added only after
# the coverage ledger shows a name is missing *and* a human confirms the vendor
# symbol. Never auto-populate -- a wrong remap silently swaps one company's
# price history for another's.
RENAME_TABLE: dict[str, str] = {}


def normalize_ticker(ticker: str) -> str:
    """Uppercase + strip; collapse internal whitespace. No vendor remap here."""
    return re.sub(r"\s+", "", str(ticker).strip().upper())


# --------------------------------------------------------------------------- #
# HTML table extraction (stdlib only -- no lxml/bs4 in the env)
# --------------------------------------------------------------------------- #

_FOOTNOTE = re.compile(r"\[[^\]]*\]")


def _clean_cell(text: str) -> str:
    """Strip footnote markers like ``[1]`` / ``[note 2]`` and collapse spaces."""
    return re.sub(r"\s+", " ", _FOOTNOTE.sub("", text)).strip()


class _TableExtractor(HTMLParser):
    """Collect every ``<table>`` as a list of rows, each a list of cell strings.

    Deliberately flat: it does not model nested tables (the S&P 500 constituents
    and changes tables do not nest tables inside the cells we read). Header and
    data cells (``th``/``td``) are both captured in row order.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(_clean_cell("".join(self._cell)))  # type: ignore[union-attr]
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._table.append(self._row)  # type: ignore[union-attr]
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def extract_tables(html: str) -> list[list[list[str]]]:
    """Parse raw HTML into a list of tables (rows of cleaned cell strings)."""
    parser = _TableExtractor()
    parser.feed(html)
    return parser.tables


def _header_index(table: list[list[str]], *needles: str) -> int | None:
    """First row index whose cells contain *all* needles (case-insensitive)."""
    wants = [n.lower() for n in needles]
    for i, row in enumerate(table):
        cells = [c.lower() for c in row]
        if all(any(w in c for c in cells) for w in wants):
            return i
    return None


def parse_constituents_table(tables: list[list[list[str]]]) -> list[str]:
    """Tickers from the constituents table (header contains 'Symbol')."""
    for table in tables:
        hi = _header_index(table, "symbol")
        if hi is None:
            continue
        header = [c.lower() for c in table[hi]]
        col = next((j for j, c in enumerate(header) if "symbol" in c), 0)
        out: list[str] = []
        for row in table[hi + 1 :]:
            if col < len(row) and row[col]:
                tok = normalize_ticker(row[col])
                if tok and tok not in out:
                    out.append(tok)
        if out:
            return out
    raise ValueError("No constituents table with a 'Symbol' column was found")


def parse_changes_table(tables: list[list[list[str]]]) -> pd.DataFrame:
    """Parse the 'Selected changes' table into ``[date, added, removed]``.

    Assumes the current Wikipedia layout: a two-row header (``Date | Added |
    Removed | Reason`` over ``Ticker | Security`` sub-columns), and data rows of
    ``[date, added_ticker, added_security, removed_ticker, removed_security,
    reason]``. Rows with an unparseable date are skipped. Column layout is
    pinned here on purpose -- if Wikipedia restructures the table this raises
    rather than silently mis-mapping tickers.
    """
    for table in tables:
        hi = _header_index(table, "added", "removed")
        if hi is None:
            continue
        # Data starts after the (one- or two-row) header block. The sub-header
        # row repeats 'Ticker'/'Security'; skip any leading rows that look like
        # headers (no parseable date in column 0).
        records: list[dict[str, object]] = []
        for row in table[hi + 1 :]:
            if not row:
                continue
            date = pd.to_datetime(row[0], errors="coerce")
            if pd.isna(date):
                continue
            added = normalize_ticker(row[1]) if len(row) > 1 else ""
            removed = normalize_ticker(row[3]) if len(row) > 3 else ""
            records.append({"date": date, "added": added, "removed": removed})
        if records:
            return pd.DataFrame.from_records(records, columns=["date", "added", "removed"])
    raise ValueError("No 'Selected changes' table with Added/Removed columns was found")


def fetch_sp500_wikipedia(url: str = WIKI_URL, timeout: float = 30.0) -> tuple[list[str], pd.DataFrame]:
    """Fetch the Wikipedia page and return ``(current_members, changes_df)``.

    Network call (the only one in this module). Kept thin so the parsing and
    reconstruction logic stays unit-testable on raw HTML / DataFrames.
    """
    import requests  # local import: keep module importable without network stack

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "trading-ensemble/universe-builder"})
    resp.raise_for_status()
    tables = extract_tables(resp.text)
    return parse_constituents_table(tables), parse_changes_table(tables)


# --------------------------------------------------------------------------- #
# Point-in-time membership reconstruction
# --------------------------------------------------------------------------- #


def reconstruct_membership(
    current_members: Iterable[str],
    changes: pd.DataFrame,
    *,
    end_date: str | pd.Timestamp,
    history_floor: pd.Timestamp = HISTORY_FLOOR,
) -> pd.DataFrame:
    """Reconstruct per-ticker membership intervals ``[start, end]`` (inclusive).

    Two passes:
      1. *Reverse* over the changes (newest first) from the current member set to
         find membership just before the earliest recorded change.
      2. *Forward* sweep from that initial set, opening an interval on each
         ``added`` event and closing one on each ``removed`` event; intervals
         still open at ``end_date`` are closed there.

    Re-additions produce multiple disjoint intervals for a ticker, which is
    correct. Inconsistent events (remove a non-member, re-add an active member)
    are skipped rather than corrupting the sweep.

    Returns a tidy frame ``[ticker, start, end]`` sorted by (ticker, start).
    """
    end_date = pd.Timestamp(end_date)
    if end_date.tz is not None:  # align with tz-naive Wikipedia change dates / HISTORY_FLOOR
        end_date = end_date.tz_localize(None)
    current = {normalize_ticker(t) for t in current_members}

    ch = changes.copy()
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
    ch["added"] = ch.get("added", "").fillna("").map(normalize_ticker)
    ch["removed"] = ch.get("removed", "").fillna("").map(normalize_ticker)

    # Pass 1: reverse to the pre-history membership set.
    members = set(current)
    for row in ch.iloc[::-1].itertuples(index=False):
        if row.added:
            members.discard(row.added)
        if row.removed:
            members.add(row.removed)
    initial = set(members)

    # Pass 2: forward sweep into intervals.
    active = set(initial)
    open_start: dict[str, pd.Timestamp] = {t: history_floor for t in initial}
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for row in ch.itertuples(index=False):
        if row.removed and row.removed in active:
            rows.append((row.removed, open_start.pop(row.removed), row.date))
            active.discard(row.removed)
        if row.added and row.added not in active:
            active.add(row.added)
            open_start[row.added] = row.date
    for ticker in active:
        rows.append((ticker, open_start[ticker], end_date))

    out = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    # Drop inverted intervals (start > end): these come from changes dated after
    # ``end_date`` -- announced-but-not-yet-effective additions, which are not
    # members as of the reconstruction horizon.
    out = out[out["start"] <= out["end"]]
    return out.sort_values(["ticker", "start"], kind="stable").reset_index(drop=True)


def ever_members(membership: pd.DataFrame) -> list[str]:
    """Sorted union of every ticker that was ever a member (survivorship-free)."""
    return sorted(set(membership["ticker"].map(normalize_ticker)))


def members_active_between(
    membership: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[str]:
    """Ever-members whose membership overlaps ``[start, end]``.

    Survivorship-free *within the window*: includes names removed during it,
    excludes only names that left before it (irrelevant to a backtest over the
    window). This is the right scoping for a price pull -- fetching names delisted
    years before ``start`` adds cost without adding in-sample breadth.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    lo = lo.tz_localize(None) if lo.tz is not None else lo
    hi = hi.tz_localize(None) if hi.tz is not None else hi
    starts = pd.to_datetime(membership["start"])
    ends = pd.to_datetime(membership["end"])
    if starts.dt.tz is not None:
        starts = starts.dt.tz_localize(None)
    if ends.dt.tz is not None:
        ends = ends.dt.tz_localize(None)
    overlap = (starts <= hi) & (ends >= lo)
    return sorted(set(membership.loc[overlap, "ticker"].map(normalize_ticker)))


def build_membership_mask(
    membership: pd.DataFrame, index: pd.Index, symbols: Iterable[str]
) -> pd.DataFrame:
    """Day x symbol boolean: was ``symbol`` an index member on that day?

    Comparison is on calendar dates (tz dropped) because membership is daily and
    the price index is tz-aware (ET); this sidesteps tz-mismatch errors while
    staying correct to the day.
    """
    idx = pd.DatetimeIndex(index)
    idx_dates = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    cols = [normalize_ticker(s) for s in symbols]
    mask = pd.DataFrame(False, index=index, columns=cols)

    by_ticker = {t: g for t, g in membership.assign(ticker=membership["ticker"].map(normalize_ticker)).groupby("ticker")}
    for ticker in cols:
        grp = by_ticker.get(ticker)
        if grp is None:
            continue
        col = np.zeros(len(idx_dates), dtype=bool)
        for r in grp.itertuples(index=False):
            start = pd.Timestamp(r.start).normalize()
            end = pd.Timestamp(r.end).normalize()
            col |= (idx_dates >= start) & (idx_dates <= end)
        mask[ticker] = col
    return mask


# --------------------------------------------------------------------------- #
# Coverage ledger (the integrity crux of free survivorship-free data)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoverageReport:
    """Honest accounting of which ever-members have usable price data."""

    asof: str
    n_ever_members: int
    n_resolved: int
    resolved: list[str]
    skipped: list[str]
    rename_table: dict[str, str]

    @property
    def coverage_fraction(self) -> float:
        return self.n_resolved / self.n_ever_members if self.n_ever_members else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "asof": self.asof,
            "n_ever_members": self.n_ever_members,
            "n_resolved": self.n_resolved,
            "n_skipped": len(self.skipped),
            "coverage_fraction": self.coverage_fraction,
            "resolved": self.resolved,
            "skipped": self.skipped,
            "rename_table": dict(self.rename_table),
        }


def compute_coverage(
    ever: Iterable[str],
    available: Iterable[str],
    *,
    asof: str,
    rename_table: dict[str, str] | None = None,
) -> CoverageReport:
    """Split the ever-member union into price-resolved vs skipped (delisted gaps).

    ``available`` is the set of symbols the price vendor actually returned bars
    for; a ticker resolves if its (renamed) symbol is in that set. The skipped
    list is the *measured* survivorship leak -- names that belong in the
    universe but have no price history. It must be surfaced, never hidden.
    """
    rename_table = rename_table or RENAME_TABLE
    avail = {normalize_ticker(s) for s in available}
    resolved, skipped = [], []
    for raw in ever:
        t = normalize_ticker(raw)
        vendor = normalize_ticker(rename_table.get(t, t))
        (resolved if vendor in avail else skipped).append(t)
    return CoverageReport(
        asof=asof,
        n_ever_members=len(resolved) + len(skipped),
        n_resolved=len(resolved),
        resolved=sorted(resolved),
        skipped=sorted(skipped),
        rename_table=dict(rename_table),
    )


def write_universe_file(symbols: Iterable[str], path: str | Path, *, asof: str, source: str = "wikipedia") -> None:
    """Write the ever-member union in the ``load_universe`` format.

    Mirrors ``data/universe/2026-05-30.txt``: a comment header documenting the
    survivorship caveat, then one symbol per line.
    """
    syms = sorted({normalize_ticker(s) for s in symbols})
    header = (
        f"# Survivorship-bias-free S&P 500 universe (union of all ever-members), as of {asof}.\n"
        f"# Source: {source} 'List of S&P 500 companies' (current + 'Selected changes').\n"
        "# One symbol per line; '#' starts a comment; blank lines ignored.\n"
        "# Includes names since removed by delisting/acquisition/relegation. Price\n"
        "# coverage for those is partial and is recorded in the companion coverage\n"
        "# JSON -- missing delisted prices are a *measured* survivorship leak, not a\n"
        "# silent drop.\n"
    )
    Path(path).write_text(header + "\n".join(syms) + "\n")
