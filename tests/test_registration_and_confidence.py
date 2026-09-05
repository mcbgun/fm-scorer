import pandas as pd
from conftest import make_player, make_squad

import parser
from parser import parse_attr_cell, parse_registration_file
from profiles import PROFILES
from registration import (
    is_hgn,
    is_u21_exempt,
    merge_registration_view,
    optimize_squad_registration,
    parse_inf_statuses,
)
from scorer import get_best_11
from season import detect_season
from squad_model import SquadAnalysis

# ---------------------------------------------------------------- U21 boundaries


def test_u21_turning_21_on_1_jan_is_not_exempt():
    # Born 1 Jan 2008 -> exactly 21 on 1 Jan 2029 -> too old for U21 (<= 20).
    assert is_u21_exempt("01/01/2008 (21)", 2029) is False


def test_u21_turning_21_the_day_after_1_jan_is_exempt():
    # Born 2 Jan 2008 -> still 20 on 1 Jan 2029.
    assert is_u21_exempt("02/01/2008 (20)", 2029) is True


def test_u21_unknown_dob_or_season_is_none():
    assert is_u21_exempt("", 2029) is None
    assert is_u21_exempt(None, 2029) is None
    assert is_u21_exempt("02/01/2008 (20)", None) is None


def test_u21_age_cutoff_is_configurable():
    assert is_u21_exempt("01/01/2008 (21)", 2029, u21_age=21) is True


# ---------------------------------------------------------------- HGN / Inf


def test_hgn_comes_from_home_grown_status_not_inf():
    assert is_hgn("Trained at club (0-21)") is True
    assert is_hgn("Trained in nation (15-21)") is True
    assert is_hgn("HGN") is False  # the Inf icon is not the source of truth
    assert is_hgn("") is False
    assert is_hgn(None) is False


def test_inf_statuses_are_split_and_preserved():
    assert parse_inf_statuses("U21 HGN Wnt") == ["U21", "HGN", "Wnt"]
    assert parse_inf_statuses("Inj") == ["Inj"]
    assert parse_inf_statuses(None) == []


def test_registration_export_merge_keeps_squad_when_name_missing():
    squad = make_squad([make_player("Alice", "GK", Inf="Spt"), make_player("Bob", "D (C)")])
    reg = pd.DataFrame({"Name": ["Bob", "Nobody"], "Inf": ["Inj", "Wnt"]})
    out = merge_registration_view(squad, reg)
    assert out.loc[out["Name"] == "Bob", "Inf"].iloc[0] == "Inj"
    assert out.loc[out["Name"] == "Alice", "Inf"].iloc[0] == "Spt"


def test_parse_registration_file_requires_inf():
    html = b"<table><tr><th>Name</th><th>Inf</th></tr><tr><td>Bob</td><td>U21</td></tr></table>"
    df = parse_registration_file(html)
    assert list(df["Inf"]) == ["U21"]


def test_html_parser_recovers_when_pandas_parser_fails(monkeypatch):
    html = b"<html><body><table><tr><th>Name</th><th>Acc</th></tr><tr><td>Bob<td>14</tr></table></body></html>"

    def fail_read_html(*args, **kwargs):
        raise RuntimeError("simulated parser failure")

    monkeypatch.setattr(parser.pd, "read_html", fail_read_html)
    df = parser._read_first_table(html)

    assert list(df["Name"]) == ["Bob"]
    assert list(df["Acc"]) == ["14"]


# ---------------------------------------------------------------- season detection


def test_detect_season_from_dob_and_age():
    df = pd.DataFrame({
        "DoB": ["15/03/2005 (24)", "20/11/2001 (27)", "02/07/2009 (20)"],
        "Age": [24, 27, 20],
    })
    info = detect_season(df)
    # 24 on 15/03/2029..14/03/2030 and 27 on 20/11/2028..19/11/2029 -> in-season 2029/30.
    assert info.start_year == 2029
    assert info.label == "2029/30"
    assert info.confident


def test_detect_season_unknown_without_dob():
    info = detect_season(pd.DataFrame({"Name": ["x"]}))
    assert info.start_year is None
    assert info.label == "Unknown season"


# ---------------------------------------------------------------- registration optimiser


def test_registration_uses_independent_hgn_and_u21():
    rows = []
    for i in range(6):
        rows.append(make_player(f"Senior{i}", "D (C)", dob="01/09/2000 (29)",
                                **{"Home-Grown Status": "Trained at club (0-21)" if i < 2 else ""}))
    rows.append(make_player("Kid", "ST (C)", dob="05/05/2010 (19)"))
    df = make_squad(rows)
    formation = [{"pos": "GK", "role": "sks"}, {"pos": "STC", "role": "afa"}]
    analysis = SquadAnalysis(df, formation, PROFILES["default"])
    res = optimize_squad_registration(df, formation, PROFILES["default"], max_squad=5, min_squad=1, min_hgn=2, analysis=analysis)
    names_u21 = {p["name"] for p in res.u21_exempt}
    assert "Kid" in names_u21
    assert res.season.start_year == 2029
    assert res.hgn_count >= 2
    assert res.constraints_met
    assert all(p["name"] != "Kid" for p in res.registered)


# ---------------------------------------------------------------- confidence


def test_parse_attr_cell_ranges():
    assert parse_attr_cell("14") == (14, 14, 14, True)
    assert parse_attr_cell("12-16") == (14, 12, 16, True)
    assert parse_attr_cell("-") == (0, 0, 0, False)
    assert parse_attr_cell(None) == (0, 0, 0, False)


def test_ranges_produce_score_bands_and_scouting_flag():
    df = make_squad([
        make_player("Known", "ST (C)", attrs={"Fin": 15}),
        make_player("Partial", "ST (C)", attrs={"Fin": "10-18", "Pac": "8-14"}),
    ])
    assert bool(df.loc[0, "Needs Scouting"]) is False
    assert bool(df.loc[1, "Needs Scouting"]) is True
    formation = [{"pos": "STC", "role": "afa"}]
    lo = get_best_11(df, formation, PROFILES["default"], assumption="low")[0]
    hi = get_best_11(df, formation, PROFILES["default"], assumption="high")[0]
    mid = get_best_11(df, formation, PROFILES["default"], assumption="mid")[0]
    assert lo["score"] <= mid["score"] <= hi["score"]
