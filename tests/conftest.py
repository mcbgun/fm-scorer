import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parser import REQUIRED_ATTRS, add_confidence_columns  # noqa: E402


def make_player(name, position, dob="01/01/2005 (24)", attrs=None, **cols):
    """Build one squad row with every required attribute set to a base value."""
    base = {a: 10 for a in REQUIRED_ATTRS}
    base.update(attrs or {})
    row = {
        "Name": name,
        "Position": position,
        "Age": int(dob.split("(")[1].rstrip(")")) if "(" in dob else 24,
        "DoB": dob,
        "Transfer Value": "£1M",
        "Wage": "£5,000 p/w",
        "Home-Grown Status": "",
        "Personality": "Balanced",
        **base,
        **cols,
    }
    return row


def make_squad(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Inf"] = df.get("Inf", pd.Series([""] * len(df)))
    return add_confidence_columns(df)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FM_SCORER_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def real_squad_path():
    p = os.environ.get("FM_SCORER_TEST_SQUAD")
    if not p or not Path(p).exists():
        pytest.skip("set FM_SCORER_TEST_SQUAD to a real Squad export to run this test")
    return Path(p)
