"""SQLite persistence: workspaces, upload snapshots, player history, shortlists
and scenarios.

The database lives in the user data directory (``paths.data_dir()``), never in
the repository. DataFrames are stored as gzip-compressed JSON (``orient=split``)
so snapshots are portable and free of pickle's code-execution risk.
"""

import base64
import gzip
import json
import sqlite3
from datetime import datetime

import pandas as pd

from paths import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    club TEXT DEFAULT '',
    season_label TEXT DEFAULT '',
    settings TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    filename TEXT DEFAULT '',
    n_rows INTEGER DEFAULT 0,
    club TEXT DEFAULT '',
    season_label TEXT DEFAULT '',
    label TEXT DEFAULT '',
    summary TEXT DEFAULT '{}',
    data BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ws ON snapshots(workspace_id, kind, created_at);
CREATE TABLE IF NOT EXISTS player_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    player_key TEXT NOT NULL,
    name TEXT NOT NULL,
    age INTEGER,
    season_label TEXT DEFAULT '',
    best_role TEXT DEFAULT '',
    best_score REAL,
    value_lo REAL,
    value_hi REAL,
    wage_k REAL,
    attrs TEXT DEFAULT '{}',
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_player ON player_history(workspace_id, player_key, recorded_at);
CREATE TABLE IF NOT EXISTS shortlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT DEFAULT '',
    meta TEXT DEFAULT '{}',
    added_at TEXT NOT NULL,
    UNIQUE(workspace_id, name, source)
);
CREATE TABLE IF NOT EXISTS scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    buys TEXT DEFAULT '[]',
    sells TEXT DEFAULT '[]',
    assumption TEXT DEFAULT 'mid',
    seasons INTEGER DEFAULT 3,
    created_at TEXT NOT NULL
);
"""

KINDS = ("squad", "targets", "registration")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def df_to_blob(df: pd.DataFrame) -> bytes:
    payload = json.loads(df.to_json(orient="split", date_format="iso", force_ascii=False, index=False))
    payload["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    return gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def blob_to_df(blob: bytes) -> pd.DataFrame:
    payload = json.loads(gzip.decompress(blob).decode("utf-8"))
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    for col, dtype in payload.get("dtypes", {}).items():
        if col in df.columns and dtype != "object":
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):
                pass
    return df


class Store:
    def __init__(self, path=None):
        self.path = str(path) if path is not None else str(data_dir() / "fm-scorer.sqlite3")
        self._cache: dict[tuple[int, str], tuple[int, pd.DataFrame]] = {}
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    # ---------------------------------------------------------------- workspaces
    def list_workspaces(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM workspaces ORDER BY updated_at DESC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["settings"] = json.loads(d["settings"] or "{}")
                d["n_snapshots"] = c.execute("SELECT COUNT(*) FROM snapshots WHERE workspace_id=?", (r["id"],)).fetchone()[0]
                out.append(d)
            return out

    def create_workspace(self, name: str, settings: dict | None = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO workspaces(name, settings, created_at, updated_at) VALUES (?,?,?,?)",
                (name, json.dumps(settings or {}), _now(), _now()),
            )
            wid = cur.lastrowid
            c.execute("INSERT OR REPLACE INTO app_state(key, value) VALUES ('active_workspace', ?)", (str(wid),))
            return wid

    def get_workspace(self, wid: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM workspaces WHERE id=?", (wid,)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["settings"] = json.loads(d["settings"] or "{}")
            return d

    def active_workspace_id(self) -> int:
        with self._conn() as c:
            r = c.execute("SELECT value FROM app_state WHERE key='active_workspace'").fetchone()
            if r and c.execute("SELECT 1 FROM workspaces WHERE id=?", (int(r["value"]),)).fetchone():
                return int(r["value"])
            first = c.execute("SELECT id FROM workspaces ORDER BY id LIMIT 1").fetchone()
        if first:
            self.set_active_workspace(first["id"])
            return first["id"]
        return self.create_workspace("My save")

    def set_active_workspace(self, wid: int) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO app_state(key, value) VALUES ('active_workspace', ?)", (str(wid),))

    def rename_workspace(self, wid: int, name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE workspaces SET name=?, updated_at=? WHERE id=?", (name, _now(), wid))

    def delete_workspace(self, wid: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM workspaces WHERE id=?", (wid,))
        self._cache = {k: v for k, v in self._cache.items() if k[0] != wid}

    def get_settings(self, wid: int) -> dict:
        ws = self.get_workspace(wid)
        return ws["settings"] if ws else {}

    def update_settings(self, wid: int, **changes) -> dict:
        settings = self.get_settings(wid)
        settings.update(changes)
        with self._conn() as c:
            c.execute("UPDATE workspaces SET settings=?, updated_at=? WHERE id=?", (json.dumps(settings, ensure_ascii=False), _now(), wid))
        return settings

    def set_workspace_meta(self, wid: int, club: str | None = None, season_label: str | None = None) -> None:
        with self._conn() as c:
            if club is not None:
                c.execute("UPDATE workspaces SET club=? WHERE id=?", (club, wid))
            if season_label is not None:
                c.execute("UPDATE workspaces SET season_label=? WHERE id=?", (season_label, wid))
            c.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (_now(), wid))

    # ---------------------------------------------------------------- snapshots
    def save_snapshot(self, wid: int, kind: str, df: pd.DataFrame, filename: str = "", summary: dict | None = None,
                      club: str = "", season_label: str = "", label: str = "") -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown snapshot kind {kind}")
        blob = df_to_blob(df)
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO snapshots(workspace_id, kind, filename, n_rows, club, season_label, label, summary, data, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (wid, kind, filename, len(df), club, season_label, label, json.dumps(summary or {}, ensure_ascii=False, default=str), blob, _now()),
            )
            sid = cur.lastrowid
            c.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (_now(), wid))
        self._cache[(wid, kind)] = (sid, df)
        return sid

    def latest_snapshot_meta(self, wid: int, kind: str) -> dict | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT id, kind, filename, n_rows, club, season_label, label, summary, created_at FROM snapshots WHERE workspace_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                (wid, kind),
            ).fetchone()
            if not r:
                return None
            d = dict(r)
            d["summary"] = json.loads(d["summary"] or "{}")
            return d

    def list_snapshots(self, wid: int, kind: str | None = None) -> list[dict]:
        with self._conn() as c:
            q = "SELECT id, kind, filename, n_rows, club, season_label, label, created_at FROM snapshots WHERE workspace_id=?"
            args: list = [wid]
            if kind:
                q += " AND kind=?"
                args.append(kind)
            q += " ORDER BY id DESC"
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def load_latest(self, wid: int, kind: str) -> pd.DataFrame | None:
        meta = self.latest_snapshot_meta(wid, kind)
        if not meta:
            return None
        cached = self._cache.get((wid, kind))
        if cached and cached[0] == meta["id"]:
            return cached[1]
        df = self.load_snapshot(meta["id"])
        if df is not None:
            self._cache[(wid, kind)] = (meta["id"], df)
        return df

    def load_snapshot(self, sid: int) -> pd.DataFrame | None:
        with self._conn() as c:
            r = c.execute("SELECT data FROM snapshots WHERE id=?", (sid,)).fetchone()
        return blob_to_df(r["data"]) if r else None

    def delete_snapshot(self, sid: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM snapshots WHERE id=?", (sid,))
        self._cache = {k: v for k, v in self._cache.items() if v[0] != sid}

    def clear_kind(self, wid: int, kind: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM snapshots WHERE workspace_id=? AND kind=?", (wid, kind))
        self._cache.pop((wid, kind), None)

    # ----------------------------------------------------------------- history
    def record_history(self, wid: int, sid: int, rows: list[dict]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT INTO player_history(workspace_id, snapshot_id, player_key, name, age, season_label, best_role, best_score, value_lo, value_hi, wage_k, attrs, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (wid, sid, r["player_key"], r["name"], r.get("age"), r.get("season_label", ""), r.get("best_role", ""),
                     r.get("best_score"), r.get("value_lo"), r.get("value_hi"), r.get("wage_k"), json.dumps(r.get("attrs", {}), ensure_ascii=False), _now())
                    for r in rows
                ],
            )

    def player_history(self, wid: int, player_key: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT h.*, s.created_at AS snapshot_at FROM player_history h JOIN snapshots s ON s.id=h.snapshot_id WHERE h.workspace_id=? AND h.player_key=? ORDER BY h.snapshot_id",
                (wid, player_key),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["attrs"] = json.loads(d["attrs"] or "{}")
                out.append(d)
            return out

    def history_overview(self, wid: int) -> list[dict]:
        """Per player: first and latest snapshot values (for trend tables)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT player_key, name, snapshot_id, age, season_label, best_role, best_score, value_lo, value_hi, wage_k FROM player_history WHERE workspace_id=? ORDER BY player_key, snapshot_id",
                (wid,),
            ).fetchall()
        by_key: dict[str, list[dict]] = {}
        for r in rows:
            by_key.setdefault(r["player_key"], []).append(dict(r))
        out = []
        for key, lst in by_key.items():
            first, last = lst[0], lst[-1]
            out.append({
                "player_key": key,
                "name": last["name"],
                "points": len(lst),
                "first_season": first["season_label"],
                "last_season": last["season_label"],
                "age": last["age"],
                "best_role": last["best_role"],
                "score_first": first["best_score"],
                "score_last": last["best_score"],
                "score_delta": round((last["best_score"] or 0) - (first["best_score"] or 0), 1),
                "value_first": first["value_lo"],
                "value_last": last["value_lo"],
                "wage_first": first["wage_k"],
                "wage_last": last["wage_k"],
            })
        out.sort(key=lambda r: r["score_delta"], reverse=True)
        return out

    # --------------------------------------------------------------- shortlist
    def shortlist(self, wid: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM shortlist WHERE workspace_id=? ORDER BY added_at DESC", (wid,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["meta"] = json.loads(d["meta"] or "{}")
                out.append(d)
            return out

    def shortlist_add(self, wid: int, name: str, source: str, note: str = "", meta: dict | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO shortlist(workspace_id, name, source, note, meta, added_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(workspace_id, name, source) DO UPDATE SET note=excluded.note, meta=excluded.meta",
                (wid, name, source, note, json.dumps(meta or {}, ensure_ascii=False, default=str), _now()),
            )

    def shortlist_remove(self, wid: int, name: str, source: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM shortlist WHERE workspace_id=? AND name=? AND source=?", (wid, name, source))

    def shortlist_names(self, wid: int, source: str | None = None) -> set[str]:
        return {s["name"] for s in self.shortlist(wid) if source is None or s["source"] == source}

    # --------------------------------------------------------------- scenarios
    def scenarios(self, wid: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM scenarios WHERE workspace_id=? ORDER BY id", (wid,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["buys"] = json.loads(d["buys"] or "[]")
                d["sells"] = json.loads(d["sells"] or "[]")
                out.append(d)
            return out

    def save_scenario(self, wid: int, name: str, buys: list[str], sells: list[str], assumption: str = "mid", seasons: int = 3, sid: int | None = None) -> int:
        with self._conn() as c:
            if sid:
                c.execute(
                    "UPDATE scenarios SET name=?, buys=?, sells=?, assumption=?, seasons=? WHERE id=? AND workspace_id=?",
                    (name, json.dumps(buys, ensure_ascii=False), json.dumps(sells, ensure_ascii=False), assumption, seasons, sid, wid),
                )
                return sid
            cur = c.execute(
                "INSERT INTO scenarios(workspace_id, name, buys, sells, assumption, seasons, created_at) VALUES (?,?,?,?,?,?,?)",
                (wid, name, json.dumps(buys, ensure_ascii=False), json.dumps(sells, ensure_ascii=False), assumption, seasons, _now()),
            )
            return cur.lastrowid

    def delete_scenario(self, wid: int, sid: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM scenarios WHERE id=? AND workspace_id=?", (sid, wid))

    # ------------------------------------------------------------ import/export
    def export_workspace(self, wid: int) -> dict:
        ws = self.get_workspace(wid)
        if not ws:
            raise KeyError(wid)
        with self._conn() as c:
            snaps = c.execute("SELECT * FROM snapshots WHERE workspace_id=? ORDER BY id", (wid,)).fetchall()
            hist = c.execute("SELECT * FROM player_history WHERE workspace_id=? ORDER BY id", (wid,)).fetchall()
        return {
            "format": "fm-scorer-workspace/1",
            "workspace": {k: ws[k] for k in ("name", "club", "season_label", "settings", "created_at")},
            "snapshots": [
                {**{k: s[k] for k in ("kind", "filename", "n_rows", "club", "season_label", "label", "summary", "created_at")},
                 "data_gz_b64": base64.b64encode(s["data"]).decode("ascii"), "old_id": s["id"]}
                for s in snaps
            ],
            "history": [dict(h) for h in hist],
            "shortlist": self.shortlist(wid),
            "scenarios": self.scenarios(wid),
        }

    def import_workspace(self, payload: dict, name: str | None = None) -> int:
        if payload.get("format") != "fm-scorer-workspace/1":
            raise ValueError("Unsupported workspace file")
        ws = payload["workspace"]
        wid = self.create_workspace(name or ws.get("name", "Imported"), ws.get("settings") or {})
        self.set_workspace_meta(wid, ws.get("club", ""), ws.get("season_label", ""))
        id_map: dict[int, int] = {}
        with self._conn() as c:
            for s in payload.get("snapshots", []):
                cur = c.execute(
                    "INSERT INTO snapshots(workspace_id, kind, filename, n_rows, club, season_label, label, summary, data, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (wid, s["kind"], s.get("filename", ""), s.get("n_rows", 0), s.get("club", ""), s.get("season_label", ""), s.get("label", ""),
                     s.get("summary", "{}") if isinstance(s.get("summary"), str) else json.dumps(s.get("summary", {})),
                     base64.b64decode(s["data_gz_b64"]), s.get("created_at", _now())),
                )
                id_map[s.get("old_id", -1)] = cur.lastrowid
            for h in payload.get("history", []):
                sid = id_map.get(h.get("snapshot_id"))
                if sid is None:
                    continue
                c.execute(
                    "INSERT INTO player_history(workspace_id, snapshot_id, player_key, name, age, season_label, best_role, best_score, value_lo, value_hi, wage_k, attrs, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (wid, sid, h["player_key"], h["name"], h.get("age"), h.get("season_label", ""), h.get("best_role", ""), h.get("best_score"),
                     h.get("value_lo"), h.get("value_hi"), h.get("wage_k"), h.get("attrs", "{}") if isinstance(h.get("attrs"), str) else json.dumps(h.get("attrs", {})), h.get("recorded_at", _now())),
                )
        for s in payload.get("shortlist", []):
            self.shortlist_add(wid, s["name"], s["source"], s.get("note", ""), s.get("meta") or {})
        for sc in payload.get("scenarios", []):
            self.save_scenario(wid, sc["name"], sc.get("buys", []), sc.get("sells", []), sc.get("assumption", "mid"), sc.get("seasons", 3))
        return wid
