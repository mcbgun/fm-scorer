"""Parsing and formatting of FM money strings ("£34M - £55M", "£19,000 p/w")."""

import re

import pandas as pd

_NUM_RE = re.compile(r"[\$£€]?\s*([\d.,]+)\s*([KMB]?)", re.IGNORECASE)


def _values_in_millions(val) -> list[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if s in ("", "-", "nan", "None", "Not for Sale", "Unknown"):
        return []
    out = []
    for num_str, unit in _NUM_RE.findall(s):
        try:
            num = float(num_str.replace(",", ""))
        except ValueError:
            continue
        unit = unit.upper()
        if unit == "K":
            num /= 1000
        elif unit == "B":
            num *= 1000
        elif unit == "":
            num /= 1_000_000
        out.append(num)
    return out


def parse_value_range(val) -> tuple[float, float]:
    """Return (low, high) transfer value in £M. Single values give low == high."""
    vals = _values_in_millions(val)
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def parse_value_low(val) -> float:
    return parse_value_range(val)[0]


def parse_value_high(val) -> float:
    return parse_value_range(val)[1]


def parse_value_mid(val) -> float:
    lo, hi = parse_value_range(val)
    return (lo + hi) / 2


def value_is_range(val) -> bool:
    lo, hi = parse_value_range(val)
    return hi > lo


def parse_wage(wage_str) -> float:
    """Parse a wage like '£19,000 p/w' or '£1.2K p/w' into weekly wage in £K."""
    if wage_str is None or (isinstance(wage_str, float) and pd.isna(wage_str)):
        return 0.0
    s = str(wage_str).strip()
    if s in ("", "-", "nan", "None"):
        return 0.0
    m = _NUM_RE.search(s)
    if not m:
        return 0.0
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return 0.0
    unit = m.group(2).upper()
    if unit == "K":
        return num
    if unit == "M":
        return num * 1000
    return num / 1000


def fmt_millions(m: float, signed: bool = False) -> str:
    """Format £M as a short string: 0.75 -> '£750K', 12.5 -> '£12.5M'."""
    sign = "-" if m < 0 else ("+" if signed and m > 0 else "")
    a = abs(m)
    if a == 0:
        return "£0"
    if a < 1:
        return f"{sign}£{a * 1000:.0f}K"
    if a >= 1000:
        return f"{sign}£{a / 1000:.2f}B"
    return f"{sign}£{a:.1f}M".replace(".0M", "M")


def fmt_wage(k_per_week: float, signed: bool = False) -> str:
    sign = "-" if k_per_week < 0 else ("+" if signed and k_per_week > 0 else "")
    a = abs(k_per_week)
    if a >= 1000:
        return f"{sign}£{a / 1000:.2f}M p/w"
    return f"{sign}£{a:.1f}K p/w".replace(".0K", "K")
