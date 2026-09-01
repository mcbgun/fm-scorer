"""HTML parser for FM24 player search/squad exports.

FM24 exports player data as HTML tables when you Ctrl+P -> Save as Web Page
using Squirrel_plays' custom views. The HTML contains a single table with
all player attributes as columns.

Expected columns (from Squirrel's views):
  Info: Inf, Name, Age, Club, Transfer Value, Wage, Nat, Position,
        Personality, Media Handling, Left Foot, Right Foot, Height
  Physical: Acc, Pac, Sta, Wor, Str, Jum, Agi, Bal
  Mental: Ant, Cnt, Dec, Tea, Pos, Vis, Agg, Bra, Cmp, OtB, Ldr
  Technical: Cro, Dri, Fin, Fir, Hea, Lon, Mar, Pas, Tck, Tec
  Goalkeeper: Aer, Cmd, Han, Kic, 1v1, Ref, TRO, Thr
"""

import io
import re

import pandas as pd


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


def _coerce_attr_column(series: pd.Series) -> pd.Series:
    """Coerce an attribute column to numeric, handling FM24 scouting ranges.

    FM24 exports partially-scouted attributes as ranges like "12-16" or
    unknown attributes as "-". This function:
      - Ranges "12-16": take the midpoint (14)
      - "-": treat as 0 (unknown)
      - Plain integers: use as-is
    """
    def _parse(val):
        if pd.isna(val):
            return 0
        s = str(val).strip()
        if s == "-" or s == "":
            return 0
        # Match range like "12-16" or "9-13"
        m = re.match(r"^(\d+)-(\d+)$", s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return (lo + hi) / 2
        try:
            return float(s)
        except ValueError:
            return 0

    return series.apply(_parse).astype(float)


def parse_html_file(file_bytes: bytes) -> pd.DataFrame:
    """Parse an FM24 HTML export into a DataFrame.

    Args:
        file_bytes: Raw bytes of the HTML file

    Returns:
        DataFrame with all player data, numeric attributes coerced to float

    Raises:
        ValueError: if the HTML can't be parsed or required columns are missing
    """
    try:
        tables = pd.read_html(io.BytesIO(file_bytes), header=0, encoding="utf-8")
    except Exception:
        try:
            tables = pd.read_html(io.BytesIO(file_bytes), header=0)
        except Exception as e:
            raise ValueError(f"Could not parse HTML file: {e}")

    if not tables:
        raise ValueError("No tables found in HTML file")

    df = tables[0]

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Coerce attribute columns to numeric (handles scouting ranges like "12-16")
    for col in REQUIRED_ATTRS:
        if col in df.columns:
            df[col] = _coerce_attr_column(df[col])

    # Check for missing required attributes — any missing will cause KeyError later
    missing = [a for a in REQUIRED_ATTRS if a not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} attribute column(s). "
            f"Make sure you're using Squirrel_plays' FM24 views. "
            f"Missing: {', '.join(missing[:10])}"
        )

    return df


def parse_csv_file(file_path: str) -> pd.DataFrame:
    """Parse a CSV file (like the tableExport files) into a DataFrame.

    This is a fallback for users who have CSV exports instead of HTML.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    for col in REQUIRED_ATTRS:
        if col in df.columns:
            df[col] = _coerce_attr_column(df[col])

    return df
