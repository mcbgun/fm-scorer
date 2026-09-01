import importlib
import json

import pandas as pd
import pytest
from conftest import make_player, make_squad
from fastapi.testclient import TestClient

from store import Store, blob_to_df, df_to_blob


def squad_html(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows).drop(columns=["Inf"], errors="ignore")
    return df.to_html(index=False).encode("utf-8")


def sample_rows():
    rows = [make_player("Keeper", "GK", dob="01/01/2000 (29)", attrs={"Ref": 15, "Han": 15, "Cmd": 14})]
    layout = [("RB", "D (R)"), ("CB1", "D (C)"), ("CB2", "D (C)"), ("LB", "D (L)"), ("DM1", "DM"), ("DM2", "DM"),
              ("RW", "AM (R)"), ("AMC", "AM (C)"), ("LW", "AM (L)"), ("ST", "ST (C)"),
              ("Kid", "AM (RL)"), ("Backup CB", "D (C)"), ("Old ST", "ST (C)")]
    for i, (n, p) in enumerate(layout):
        dob = "05/05/2010 (19)" if n == "Kid" else ("01/02/1995 (34)" if n == "Old ST" else "01/01/2002 (27)")
        rows.append(make_player(n, p, dob=dob, attrs={"Pac": 12 + i % 4, "Acc": 12 + i % 3},
                                **{"Home-Grown Status": "Trained at club (0-21)" if i % 3 == 0 else ""}))
    return rows


# ---------------------------------------------------------------- store


def test_blob_roundtrip_preserves_dtypes_and_values():
    df = make_squad(sample_rows())
    df["All Missing"] = float("nan")
    back = blob_to_df(df_to_blob(df))
    assert list(back.columns) == list(df.columns)
    assert (back.isna().values == df.isna().values).all()
    assert (back.fillna("x").astype(str).values == df.fillna("x").astype(str).values).all()
    assert str(back["All Missing"].dtype) == "float64"


def test_store_workspaces_snapshots_history_export(tmp_data_dir):
    s = Store(tmp_data_dir / "t.db")
    wid = s.active_workspace_id()
    df = make_squad(sample_rows())
    sid = s.save_snapshot(wid, "squad", df, filename="a.html", season_label="2029/30")
    assert s.load_latest(wid, "squad").shape == df.shape
    s.record_history(wid, sid, [{"player_key": "keeper", "name": "Keeper", "age": 29, "season_label": "2029/30",
                                 "best_role": "sks", "best_score": 10.0, "value_lo": 1.0, "value_hi": 1.0,
                                 "wage_k": 5.0, "attrs": {"Ref": 15}}])
    assert s.player_history(wid, "keeper")[0]["attrs"] == {"Ref": 15}

    s.shortlist_add(wid, "Someone", "target", note="hi")
    assert s.shortlist_names(wid) == {"Someone"}
    sc = s.save_scenario(wid, "Plan A", ["Someone"], ["Old ST"])
    assert s.scenarios(wid)[0]["id"] == sc

    payload = s.export_workspace(wid)
    payload = json.loads(json.dumps(payload))  # must be JSON serialisable
    wid2 = s.import_workspace(payload, name="Copy")
    assert wid2 != wid
    assert s.load_latest(wid2, "squad").shape == df.shape
    assert s.shortlist_names(wid2) == {"Someone"}
    assert len(s.scenarios(wid2)) == 1

    s.delete_workspace(wid2)
    assert all(w["id"] != wid2 for w in s.list_workspaces())


# ---------------------------------------------------------------- app


@pytest.fixture
def client(tmp_data_dir):
    import main

    importlib.reload(main)
    return TestClient(main.app, follow_redirects=False)


def test_app_end_to_end(client):
    r = client.get("/")
    assert r.status_code == 200 and "Get started" in r.text  # empty workspace shows uploads inline

    rows = sample_rows()
    r = client.post("/upload", files={"squad_file": ("Squad.html", squad_html(rows), "text/html")},
                    data={"transfer_budget": "10", "wage_budget": "5"})
    assert r.status_code == 200 and "Loaded" in r.text

    # targets: one clear upgrade at ST, one unscouted
    targets = [make_player("Star Striker", "ST (C)", attrs={"Fin": 18, "Pac": 17, "Acc": 17, "OtB": 17, "Cmp": 16},
                           **{"Transfer Value": "£2M - £4M", "Club": "Elsewhere"}),
               make_player("Mystery", "ST (C)", attrs={"Fin": "10-18", "Pac": "10-18"}, **{"Transfer Value": "£1M", "Club": "Far"})]
    r = client.post("/upload", files={"targets_file": ("targets.html", squad_html(targets), "text/html")})
    assert r.status_code == 200

    for path, needle in [
        ("/", "Best XI"), ("/squad", "Depth chart"), ("/compare", "Star Striker"), ("/youth", "Kid"),
        ("/sell", "Recommendation"), ("/registration", "Home-grown"), ("/strategy", "Reasoning"),
        ("/player/squad/0", "Score breakdown"), ("/player/targets/1", "Scouting required"),
        ("/shortlist", "Shortlist"), ("/scenarios", "Plan A"), ("/history", "Squad snapshots (1)"),
        ("/config", "Formation editor"), ("/squad.csv", "Name"), ("/compare.csv", "Star Striker"),
    ]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert needle in r.text, path

    # exclude-unscouted hides Mystery from targets
    assert "Mystery" in client.get("/compare").text
    client.post("/settings", data={"next": "/compare", "exclude_unscouted_present": "1", "exclude_unscouted": "1"})
    assert "Mystery" not in client.get("/compare").text

    # shortlist + scenario + api
    assert client.post("/shortlist/add", data={"name": "Star Striker", "source": "target"}).status_code == 303
    assert "Star Striker" in client.get("/shortlist.csv").text
    r = client.get("/api/scenario", params={"buys": "Star Striker", "sells": "Old ST"})
    body = r.json()
    assert body["improvement"] > 0
    assert client.post("/scenarios", data={"name": "Plan A", "buys": "Star Striker", "sells": "Old ST"}).status_code == 303
    assert "Plan A" in client.get("/scenarios").text

    # custom formation + role via config, then delete
    slots = json.dumps([{"pos": "GK", "role": "sks"}] + [{"pos": p, "role": r} for p, r in [
        ("DR", "wbs"), ("DCR", "cdd"), ("DCL", "cdd"), ("DL", "wbs"), ("DMCR", "dmd"), ("DMCL", "dmd"),
        ("AMR", "ifa"), ("AMC", "aps"), ("AML", "ifa"), ("STC", "afa")]])
    assert client.post("/config/formation", data={"slots_json": slots, "save_as": "t433", "name": "Test"}).status_code == 303
    assert "Test" in client.get("/").text
    assert client.post("/config/role", data={"role_id": "zz", "name": "ZZ", "key": "Pac", "green": "Acc"}).status_code == 303
    assert "zz" in client.get("/config/export").json()["roles"]
    client.post("/config/role/delete", data={"role_id": "zz"})
    client.post("/config/formation/delete", data={"formation_id": "t433"})

    # workspace export/import + second snapshot -> history
    exported = client.get("/workspace/export").content
    assert client.post("/workspace/import", files={"file": ("ws.json", exported, "application/json")}, data={"name": "Imp"}).status_code == 303
    assert "Imp" in client.get("/").text
    client.post("/upload", files={"squad_file": ("Squad2.html", squad_html(rows), "text/html")})
    assert "Squad snapshots (2)" in client.get("/history").text


def test_upload_rejects_bad_file(client):
    r = client.post("/upload", files={"squad_file": ("x.html", b"<html><body>nope</body></html>", "text/html")})
    assert r.status_code == 200
    assert "Missing" in r.text or "Could not" in r.text or "No tables" in r.text
