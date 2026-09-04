"""HTML parser for FM24 player search/squad exports.

FM24 exports player data as HTML tables when you Ctrl+P -> Save as Web Page
using Squirrel_plays' custom views. The HTML contains a single table with
all player attributes as columns.

Scouting ranges ("12-16") and unknowns ("-") are preserved as separate low /
high columns (``Acc_lo`` / ``Acc_hi``) alongside the midpoint used for the
headline score, and a per-player ``Scouting %`` completeness figure is added so
every downstream view can show confidence rather than false precision.
"""

import io
import re

import pandas as pd

from roles import ALL_ATTRIBUTES

# All attribute columns the scorer expects
REQUIRED_ATTRS = [
    # Physical
    "Acc", "Pac", "Sta", "Wor", "Str", "Jum", "Agi", "Bal",
    # Mental
    "Ant", "Cnt", "Dec", "Tea", "Pos", "Vis", "Agg", "Bra", "Cmp", "OtB",
    # Technical
    "Cro", "Dri", "Fin", "Fir", "Hea", "Lon", "Mar", "Pas", "Tck", "Tec", "Fla",
    # Goalkeeper
    "Aer", "Cmd", "Han", "Kic", "1v1", "Ref", "TRO", "Thr",
]

# Info columns we want to keep in the output
INFO_COLS = [
    "Inf", "Name", "Age", "Club", "Transfer Value", "Wage", "Nat",
    "Position", "Personality", "Media Handling", "Left Foot", "Right Foot",
    "Height", "Home-Grown Status", "DoB",
]

# Optional columns the analysis uses when present (FM "Development"/"Contract" views).
OPTIONAL_SIGNAL_COLS = [
    "Potential", "PoTe", "PoTa", "Trn Rat", "Injury Susceptibility", "Injury Risk",
    "Playing Time", "Agreed Playing Time", "Actual Playing Time", "Expires", "Av Rat",
    "Best Pos", "Best Role", "Sec. Position", "Determination", "Det", "Reg", "UID", "Club",
]

LO_SUFFIX = "_lo"
HI_SUFFIX = "_hi"

_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_UNKNOWN = ("", "-", "nan", "NaN", "None")


def parse_attr_cell(val) -> tuple[float, float, float, bool]:
    """Parse one attribute cell into (mid, lo, hi, known).

    - "14"     -> (14, 14, 14, True)
    - "12-16"  -> (14, 12, 16, True)   (partially scouted)
    - "-" / "" -> (0, 0, 0, False)     (unknown / unscouted)
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0, 0.0, 0.0, False
    s = str(val).strip()
    if s in _UNKNOWN:
        return 0.0, 0.0, 0.0, False
    m = _RANGE_RE.match(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2, lo, hi, True
    try:
        f = float(s)
    except ValueError:
        return 0.0, 0.0, 0.0, False
    return f, f, f, True


def _coerce_attr_column(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    parsed = [parse_attr_cell(v) for v in series.tolist()]
    mid = pd.Series([p[0] for p in parsed], index=series.index, dtype=float)
    lo = pd.Series([p[1] for p in parsed], index=series.index, dtype=float)
    hi = pd.Series([p[2] for p in parsed], index=series.index, dtype=float)
    known = pd.Series([p[3] for p in parsed], index=series.index, dtype=bool)
    return mid, lo, hi, known


def fix_mojibake(text: str) -> str:
    """FM writes UTF-8 but declares another charset; '£' shows up as 'Â£'."""
    return text.replace("Â£", "£").replace("â‚¬", "€").replace("Ã©", "é")


def add_confidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<attr>_lo`` / ``<attr>_hi`` columns and ``Scouting %`` / ``Range Width``.

    Idempotent: columns that already exist are left alone.
    """
    df = df.copy()
    attrs = [a for a in ALL_ATTRIBUTES if a in df.columns]
    known_total = pd.Series(0.0, index=df.index)
    width_total = pd.Series(0.0, index=df.index)
    new_cols: dict[str, pd.Series] = {}
    n = 0
    for col in attrs:
        if f"{col}{LO_SUFFIX}" in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lo = df[f"{col}{LO_SUFFIX}"].astype(float)
            hi = df[f"{col}{HI_SUFFIX}"].astype(float)
            known = df[col].astype(float) > 0
        else:
            mid, lo, hi, known = _coerce_attr_column(df[col])
            df[col] = mid
            new_cols[f"{col}{LO_SUFFIX}"] = lo
            new_cols[f"{col}{HI_SUFFIX}"] = hi
        if col in REQUIRED_ATTRS:
            known_total = known_total + known.astype(float)
            width_total = width_total + (hi - lo)
            n += 1
    if n:
        new_cols["Scouting %"] = (known_total / n * 100).round(0)
        new_cols["Range Width"] = (width_total / n).round(1)
    else:
        new_cols["Scouting %"] = pd.Series(0.0, index=df.index)
        new_cols["Range Width"] = pd.Series(0.0, index=df.index)
    new_cols["Needs Scouting"] = (new_cols["Scouting %"] < 100) | (new_cols["Range Width"] > 0)
    df = df.drop(columns=[c for c in new_cols if c in df.columns])
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _read_first_table(file_bytes: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Could not decode HTML file")
    text = fix_mojibake(text)

    try:
        tables = pd.read_html(io.StringIO(text), header=0)
    except Exception as first_error:
        try:
            tables = pd.read_html(io.StringIO(text), header=0, flavor="html5lib")
        except Exception as fallback_error:
            raise ValueError(f"Could not parse HTML file: {fallback_error}") from first_error

    if not tables:
        raise ValueError("No tables found in HTML file")

    df = tables[0]
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    if "Name" not in df.columns:
        raise ValueError("Missing 'Name' column")
    df = df[df["Name"].notna()].reset_index(drop=True)
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def parse_registration_file(file_bytes: bytes) -> pd.DataFrame:
    """Parse the small FM registration view (Name + Inf status icons)."""
    df = _read_first_table(file_bytes)
    if "Inf" not in df.columns:
        raise ValueError("Registration export must include the 'Inf' column")
    df["Inf"] = df["Inf"].fillna("").astype(str).str.strip()
    return df


def parse_html_file(file_bytes: bytes) -> pd.DataFrame:
    """Parse an FM24 HTML export into a DataFrame.

    Raises:
        ValueError: if the HTML can't be parsed or required columns are missing
    """
    df = _read_first_table(file_bytes)

    missing = [a for a in REQUIRED_ATTRS if a not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} attribute column(s). "
            f"Make sure you're using Squirrel_plays' FM24 views. "
            f"Missing: {', '.join(missing[:10])}"
        )
    for col in ("Transfer Value", "Wage", "Asking Price"):
        if col in df.columns:
            df[col] = df[col].astype(str).map(fix_mojibake)

    return add_confidence_columns(df)


def parse_csv_file(file_path: str) -> pd.DataFrame:
    """Parse a CSV file (like the tableExport files) into a DataFrame."""
    df = pd.read_csv(file_path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    return add_confidence_columns(df)


def summarize_upload(df: pd.DataFrame, kind: str) -> dict:
    """Human-readable validation summary shown after an upload."""
    clubs = df["Club"].dropna().astype(str).value_counts() if "Club" in df.columns else pd.Series(dtype=int)
    optional_present = [c for c in OPTIONAL_SIGNAL_COLS if c in df.columns]
    optional_missing = [c for c in ("Potential", "Trn Rat", "Injury Susceptibility", "Playing Time", "Expires") if c not in df.columns]
    fully_scouted = int((df["Scouting %"] >= 100).sum()) if "Scouting %" in df.columns else len(df)
    ranged = int((df["Range Width"] > 0).sum()) if "Range Width" in df.columns else 0
    return {
        "kind": kind,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "club": str(clubs.index[0]) if len(clubs) == 1 or (len(clubs) and kind == "squad") else None,
        "clubs": int(len(clubs)),
        "duplicate_names": int(df["Name"].duplicated().sum()) if "Name" in df.columns else 0,
        "fully_scouted": fully_scouted,
        "partially_scouted": int(len(df)) - fully_scouted,
        "ranged_attributes": ranged,
        "optional_present": optional_present,
        "optional_missing": optional_missing,
    }
