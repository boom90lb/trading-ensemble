"""Universe loading + point-in-time as-of filtering (Phase 4 §4.3)."""

import pandas as pd

from scripts.training import filter_universe_asof, load_universe
from src.data_loader import BAR_TZ


def _frame(start: str, n: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, tz=BAR_TZ)
    return pd.DataFrame({"close": range(n)}, index=idx)


def test_load_universe_parses_comments_blanks_dups(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text(
        "# header comment\n"
        "AAPL\n"
        "MSFT  # inline comment\n"
        "\n"
        "GOOG\n"
        "AAPL\n"  # duplicate -> dropped
    )
    assert load_universe(str(f)) == ["AAPL", "MSFT", "GOOG"]


def test_filter_universe_asof_drops_late_listings():
    data = {
        "OLD": _frame("2018-01-02"),   # listed well before as-of
        "NEW": _frame("2025-01-02"),   # IPO after as-of -> dropped
    }
    kept = filter_universe_asof(data, "2020-01-01")
    assert set(kept) == {"OLD"}


def test_filter_universe_asof_keeps_on_boundary():
    data = {"SYM": _frame("2020-01-01")}
    # First bar exactly at the as-of date counts as investable.
    assert set(filter_universe_asof(data, "2020-01-01")) == {"SYM"}


def test_filter_universe_asof_none_is_passthrough():
    data = {"A": _frame("2020-01-02"), "B": _frame("2025-01-02")}
    assert filter_universe_asof(data, None) is data


def test_filter_universe_asof_drops_empty_frame():
    data = {"EMPTY": pd.DataFrame({"close": []}), "OK": _frame("2018-01-02")}
    assert set(filter_universe_asof(data, "2020-01-01")) == {"OK"}
