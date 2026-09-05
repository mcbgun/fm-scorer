"""Detect the in-game season from an FM export.

FM does not export the current date, but two things pin it down:

1. ``DoB`` cells look like ``30/12/2001 (28 years old)``. For each player the
   current date lies in ``[DoB + age years, DoB + age + 1 years)``. Intersecting
   those windows over a whole squad usually narrows the date to a few days.
2. Contract views carry ``Season 2031/32`` style columns for *future* seasons;
   the earliest one minus one year is the current season.

The season start year is what registration rules need: a player is U21 if born
on/after 1 January of ``season_start_year - 21``.
"""

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

_DOB_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})\s*(?:\((\d+)\s*years? old\))?")
_SEASON_COL_RE = re.compile(r"Season\s+(\d{4})/(\d{2,4})")


@dataclass
class SeasonInfo:
    start_year: int | None
    date_low: date | None = None
    date_high: date | None = None
    source: str = "unknown"
    players_used: int = 0

    @property
    def label(self) -> str:
        if self.start_year is None:
            return "Unknown season"
        return f"{self.start_year}/{str(self.start_year + 1)[-2:]}"

    @property
    def confident(self) -> bool:
        return self.start_year is not None and self.source.startswith("dob")

    def to_dict(self) -> dict:
        return {
            "start_year": self.start_year,
            "label": self.label,
            "date_low": self.date_low.isoformat() if self.date_low else None,
            "date_high": self.date_high.isoformat() if self.date_high else None,
            "source": self.source,
            "players_used": self.players_used,
        }


def parse_dob(dob_str) -> tuple[date | None, int | None]:
    """Return (birth_date, exported_age) from a DoB cell."""
    if dob_str is None or (isinstance(dob_str, float) and pd.isna(dob_str)):
        return None, None
    m = _DOB_RE.search(str(dob_str))
    if not m:
        return None, None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        birth = date(y, mth, d)
    except ValueError:
        return None, None
    age = int(m.group(4)) if m.group(4) else None
    return birth, age


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + years, day=28)


def _season_from_date(d: date) -> int:
    # FM seasons roll over in the summer; a July-December date is the start of that season.
    return d.year if d.month >= 7 else d.year - 1


def detect_season(df: pd.DataFrame) -> SeasonInfo:
    low: date | None = None
    high: date | None = None
    used = 0
    if "DoB" in df.columns:
        age_col = df["Age"] if "Age" in df.columns else None
        for i, val in enumerate(df["DoB"].tolist()):
            birth, age = parse_dob(val)
            if birth is None:
                continue
            if age is None and age_col is not None:
                try:
                    age = int(age_col.iloc[i])
                except (TypeError, ValueError):
                    age = None
            if age is None:
                continue
            lo = _add_years(birth, age)
            hi = _add_years(birth, age + 1)
            if low is None or lo > low:
                low = lo
            if high is None or hi < high:
                high = hi
            used += 1
        if low is not None and high is not None and low >= high:
            # Inconsistent data (e.g. mixed export dates) — fall back to the majority window.
            low, high, used = None, None, 0

    contract_year = None
    for col in df.columns:
        m = _SEASON_COL_RE.match(str(col))
        if m:
            y = int(m.group(1))
            contract_year = y if contract_year is None else min(contract_year, y)
    if contract_year is not None:
        contract_year -= 1  # columns list future seasons

    if low is not None and high is not None:
        start = _season_from_date(low)
        if _season_from_date(high) != start and contract_year is not None:
            start = contract_year
        source = "dob+contract" if contract_year == start else "dob"
        return SeasonInfo(start, low, high, source, used)
    if contract_year is not None:
        return SeasonInfo(contract_year, None, None, "contract_columns", 0)
    return SeasonInfo(None)
