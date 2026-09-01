"""Development signals exported by FM (when the view includes them).

FM exposes several columns that say far more about a youngster's future than
age + personality alone. All of them are optional; every helper here degrades
gracefully to a neutral factor when a column is missing or blank.

  Potential / PoTe / PoTa   coach potential rating (stars or numeric)
  Trn Rat                   training rating, 1-10 scale (7+ is good)
  Det                       Determination attribute
  Playing Time / Agreed Playing Time / Actual Playing Time
  Injury Susceptibility / Injury Risk
  Expires                   contract expiry date

The output is always a multiplier around 1.0 plus a short, player-specific
explanation string so recommendations can say *why*.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

from season import parse_dob

_STAR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?")

INJURY_AVAILABILITY = {
    "very low": 1.02,
    "low": 1.0,
    "medium": 0.96,
    "fairly high": 0.92,
    "high": 0.88,
    "very high": 0.82,
}

PLAYING_TIME_GROWTH = {
    "star player": 1.10,
    "important player": 1.10,
    "regular starter": 1.10,
    "squad player": 1.02,
    "impact sub": 0.98,
    "fringe player": 0.92,
    "breakthrough prospect": 1.0,
    "hot prospect": 1.0,
    "youngster": 0.95,
    "b team regular": 0.9,
    "surplus to requirements": 0.85,
    "emergency backup": 0.9,
}


def _cell(row, col):
    if col not in row.index:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if s in ("", "-", "nan") else s


@dataclass
class DevSignals:
    potential_stars: float | None = None  # 0.5..5 scale when exported
    potential_stars_hi: float | None = None
    training_rating: float | None = None
    determination: float | None = None
    playing_time: str | None = None
    injury: str | None = None
    contract_expiry_year: int | None = None
    notes: list[str] = field(default_factory=list)

    def growth_multiplier(self) -> float:
        m = 1.0
        if self.training_rating is not None:
            # 7.0 is an average rating; each point is worth ~8% growth.
            m *= max(0.7, min(1.3, 1.0 + (self.training_rating - 7.0) * 0.08))
        if self.determination is not None:
            m *= max(0.85, min(1.15, 1.0 + (self.determination - 12.0) * 0.02))
        if self.playing_time:
            m *= PLAYING_TIME_GROWTH.get(self.playing_time.lower(), 1.0)
        return round(m, 3)

    def availability(self) -> float:
        if not self.injury:
            return 1.0
        return INJURY_AVAILABILITY.get(self.injury.lower(), 1.0)

    def to_dict(self) -> dict:
        return {
            "potential_stars": self.potential_stars,
            "potential_stars_hi": self.potential_stars_hi,
            "training_rating": self.training_rating,
            "determination": self.determination,
            "playing_time": self.playing_time,
            "injury": self.injury,
            "contract_expiry_year": self.contract_expiry_year,
            "growth_multiplier": self.growth_multiplier(),
            "availability": self.availability(),
            "notes": self.notes,
        }


def _parse_stars(val: str | None) -> tuple[float | None, float | None]:
    if not val:
        return None, None
    m = _STAR_RE.search(val)
    if not m:
        return None, None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    if lo > 5 and lo <= 200:  # a raw PA number (1-200) slipped through
        lo, hi = lo / 40, hi / 40
    return lo, hi


def extract_signals(row: pd.Series) -> DevSignals:
    sig = DevSignals()
    for col in ("Potential", "PoTe", "PoTa"):
        lo, hi = _parse_stars(_cell(row, col))
        if lo is not None:
            sig.potential_stars, sig.potential_stars_hi = lo, hi
            band = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
            sig.notes.append(f"coach potential {band}★")
            break
    tr = _cell(row, "Trn Rat")
    if tr is not None:
        try:
            sig.training_rating = float(tr)
            if sig.training_rating >= 7.5:
                sig.notes.append(f"training well ({sig.training_rating:.2f})")
            elif sig.training_rating < 6.5:
                sig.notes.append(f"poor training rating ({sig.training_rating:.2f})")
        except ValueError:
            pass
    det = _cell(row, "Det")
    if det is not None:
        try:
            sig.determination = float(det)
        except ValueError:
            pass
    for col in ("Playing Time", "Agreed Playing Time", "Actual Playing Time"):
        pt = _cell(row, col)
        if pt:
            sig.playing_time = pt
            break
    for col in ("Injury Susceptibility", "Injury Risk"):
        inj = _cell(row, col)
        if inj:
            sig.injury = inj
            if inj.lower() in ("high", "very high", "fairly high"):
                sig.notes.append(f"{inj.lower()} injury susceptibility")
            break
    exp = _cell(row, "Expires")
    if exp:
        d, _ = parse_dob(exp)
        if d:
            sig.contract_expiry_year = d.year
        else:
            m = re.search(r"(\d{4})", exp)
            if m:
                sig.contract_expiry_year = int(m.group(1))
    return sig


def star_potential_ceiling(current_score: float, stars: float | None) -> float | None:
    """Very rough conversion of a coach potential star rating into a role
    score ceiling. Stars are relative to the club's level, so this is only a
    hint: 5★ ≈ +25% on the current score, 3★ ≈ +5%, 1★ ≈ -10%."""
    if stars is None:
        return None
    pct = -0.10 + (stars - 1.0) * 0.0875  # 1★ -> -10%, 5★ -> +25%
    return round(current_score * (1 + pct), 1)
