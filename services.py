"""Service layer between the FastAPI routes and the analysis modules.

``WorkspaceContext`` loads the active workspace (settings + latest snapshots)
and lazily builds the shared ``SquadAnalysis`` that every page reads from, so
the dashboard, Best XI, depth, sell, registration and strategy views all agree.
"""

from dataclasses import dataclass, field

import pandas as pd

from development import extract_signals
from formations import default_formation, get_formation, validate_slots
from money import fmt_millions, fmt_wage, parse_value_range, parse_wage
from parser import parse_html_file, parse_registration_file, summarize_upload
from positions import parse_slot_position, position_familiarity
from profiles import PROFILES, effective_role, get_profile
from projection import project_band
from registration import RegistrationResult, merge_registration_view, optimize_squad_registration, registration_season
from roles import ROLES
from scorer import ASSUMPTIONS, filter_formation_upgrades, score_breakdown
from season import SeasonInfo, parse_dob
from sell import build_depth_chart, generate_sell_recommendations
from squad_model import SquadAnalysis
from store import Store
from strategy import evaluate_plan, generate_strategy
from youth import analyze_youth, compute_potential_band

DEFAULT_SETTINGS = {
    "profile_id": None,          # None -> the formation's suggested profile
    "formation_id": "4231gp",
    "slots": None,               # custom slots override formation_id
    "assumption": "mid",
    "transfer_budget": None,     # £M; None = not entered
    "wage_budget": None,         # £K p/w spare; None = not entered
    "seasons": 3,
    "max_transfers": 6,
    "min_gain": 0.3,
    "min_gain_per_m": 0.05,
    "exclude_unscouted": False,
    "locked_players": [],
    "squad_size_limit": 25,
    "max_youth_age": 21,
    "max_squad": 25,
    "min_hgn": 8,
    "u21_age": 20,
    "min_margin": 0.0,
    "max_age": 99,
    "max_value": "",
    "top_n": 50,
    "position_mode": "can_play",
    "board_sales_percentage": 100.0,
}

# Pitch coordinates (percent from left / top) for the dashboard pitch graphic.
_LINE_Y = {"GK": 92, "D": 76, "WB": 70, "DM": 58, "M": 46, "AM": 30, "ST": 12}
_SIDE_X = {"L": 18, "C": 50, "R": 82, "LC": 34, "RC": 66, "": 50}


def pitch_xy(slot_pos: str, siblings: int = 1, index: int = 0) -> tuple[float, float]:
    ptype, side = parse_slot_position(slot_pos)
    y = _LINE_Y.get(ptype or "M", 46)
    if side in ("", "C") and siblings > 1:
        # spread central players of the same line evenly
        span = 30 if siblings == 2 else 44
        x = 50 - span / 2 + span * index / (siblings - 1)
    elif side in ("CR", "RC"):
        x = 62 if siblings <= 2 else 66
    elif side in ("CL", "LC"):
        x = 38 if siblings <= 2 else 34
    else:
        x = _SIDE_X.get(side, 50)
    return x, y


def pitch_layout(formation: list[dict]) -> list[dict]:
    """Return x/y for every slot, spreading same-line central slots."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(formation):
        ptype, side = parse_slot_position(s["pos"])
        if side in ("", "C"):
            groups.setdefault(ptype or "M", []).append(i)
    out = []
    for i, s in enumerate(formation):
        ptype, side = parse_slot_position(s["pos"])
        sib = groups.get(ptype or "M", [i]) if side in ("", "C") else [i]
        x, y = pitch_xy(s["pos"], len(sib), sib.index(i) if i in sib else 0)
        out.append({"x": round(x, 1), "y": y})
    return out


@dataclass
class UploadReport:
    kind: str
    filename: str
    summary: dict
    season: dict | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


class WorkspaceContext:
    def __init__(self, store: Store, wid: int | None = None):
        self.store = store
        self.wid = wid if wid is not None else store.active_workspace_id()
        self.workspace = store.get_workspace(self.wid) or {}
        self.settings = {**DEFAULT_SETTINGS, **self.workspace.get("settings", {})}
        self._squad: pd.DataFrame | None = None
        self._targets: pd.DataFrame | None = None
        self._registration: pd.DataFrame | None = None
        self._analysis: SquadAnalysis | None = None
        self._season: SeasonInfo | None = None

    # ----------------------------------------------------------------- data
    @property
    def squad(self) -> pd.DataFrame | None:
        if self._squad is None:
            raw = self.store.load_latest(self.wid, "squad")
            if raw is not None:
                self._squad = merge_registration_view(raw, self.registration)
        return self._squad

    @property
    def targets(self) -> pd.DataFrame | None:
        if self._targets is None:
            self._targets = self.store.load_latest(self.wid, "targets")
        return self._targets

    @property
    def registration(self) -> pd.DataFrame | None:
        if self._registration is None:
            self._registration = self.store.load_latest(self.wid, "registration")
        return self._registration

    @property
    def has_squad(self) -> bool:
        return self.store.latest_snapshot_meta(self.wid, "squad") is not None

    @property
    def has_targets(self) -> bool:
        return self.store.latest_snapshot_meta(self.wid, "targets") is not None

    @property
    def season(self) -> SeasonInfo:
        if self._season is None:
            if self.squad is not None:
                self._season = registration_season(self.squad, self.registration)
            else:
                self._season = SeasonInfo(None)
        return self._season

    # ------------------------------------------------------------- settings
    @property
    def formation(self) -> dict:
        slots = self.settings.get("slots")
        if slots and not validate_slots(slots):
            return {"id": "custom", "name": "Custom formation", "slots": slots, "profile": self.settings.get("profile_id")}
        f = get_formation(self.settings.get("formation_id") or "") or default_formation()
        return f

    @property
    def slots(self) -> list[dict]:
        return self.formation["slots"]

    @property
    def profile_id(self) -> str:
        pid = self.settings.get("profile_id") or self.formation.get("profile") or "default"
        return pid if pid in PROFILES else "default"

    @property
    def profile(self):
        return get_profile(self.profile_id)

    @property
    def assumption(self) -> str:
        a = self.settings.get("assumption", "mid")
        return a if a in ASSUMPTIONS else "mid"

    def save_settings(self, **changes) -> None:
        self.settings = {**DEFAULT_SETTINGS, **self.store.update_settings(self.wid, **changes)}
        self._analysis = None

    # ------------------------------------------------------------- analysis
    @property
    def analysis(self) -> SquadAnalysis | None:
        if self._analysis is None and self.squad is not None:
            self._analysis = SquadAnalysis(self.squad, self.slots, self.profile, self.assumption, season=self.season)
        return self._analysis

    def header(self) -> dict:
        """State shown in the global header on every page."""
        squad_meta = self.store.latest_snapshot_meta(self.wid, "squad")
        targets_meta = self.store.latest_snapshot_meta(self.wid, "targets")
        club = (squad_meta or {}).get("club") or self.workspace.get("club") or ""
        tb = self.settings.get("transfer_budget")
        wb = self.settings.get("wage_budget")
        return {
            "workspace": self.workspace,
            "workspaces": self.store.list_workspaces(),
            "club": club,
            "season": self.season.label if squad_meta else "",
            "season_confident": self.season.confident if squad_meta else False,
            "profile": self.profile,
            "formation": self.formation,
            "assumption": self.assumption,
            "has_squad": squad_meta is not None,
            "has_targets": targets_meta is not None,
            "has_registration": self.store.latest_snapshot_meta(self.wid, "registration") is not None,
            "squad_rows": (squad_meta or {}).get("n_rows", 0),
            "target_rows": (targets_meta or {}).get("n_rows", 0),
            "transfer_budget": tb,
            "wage_budget": wb,
            "budget_label": fmt_millions(tb) if tb is not None else "not set",
            "wage_label": fmt_wage(wb) if wb is not None else "not set",
            "shortlist_count": len(self.store.shortlist(self.wid)),
        }

    # -------------------------------------------------------------- uploads
    def ingest(self, kind: str, file_bytes: bytes, filename: str) -> UploadReport:
        try:
            df = parse_registration_file(file_bytes) if kind == "registration" else parse_html_file(file_bytes)
        except ValueError as e:
            return UploadReport(kind=kind, filename=filename, summary={}, error=str(e))
        if kind == "registration":
            summary = {"kind": kind, "rows": int(len(df)), "columns": int(len(df.columns)),
                       "statuses": sorted({t for v in df["Inf"] for t in str(v).split() if t})}
            season = None
        else:
            summary = summarize_upload(df, kind)
            season = registration_season(df).to_dict() if kind == "squad" else None
        warnings = []
        if kind == "squad" and summary.get("clubs", 1) > 1:
            warnings.append(f"Squad export contains {summary['clubs']} clubs — did you export the right view?")
        if kind == "squad" and season and not season.get("start_year"):
            warnings.append("Could not detect the season from the export (no DoB/Age pairs); U21 exemptions will be unknown.")
        if kind == "targets" and summary.get("partially_scouted", 0):
            warnings.append(f"{summary['partially_scouted']} of {summary['rows']} targets are not fully scouted — their scores are ranges, not points.")
        if kind in ("squad", "targets") and summary.get("optional_missing"):
            warnings.append("Optional columns not exported (projections will be less informed): " + ", ".join(summary["optional_missing"]))
        sid = self.store.save_snapshot(
            self.wid, kind, df, filename=filename, summary=summary,
            club=summary.get("club") or "", season_label=(season or {}).get("label", ""),
        )
        if kind == "squad":
            self.store.set_workspace_meta(self.wid, club=summary.get("club") or None, season_label=(season or {}).get("label"))
            self._squad = None
            self._season = None
            self._analysis = None
            self._record_history(sid)
        elif kind == "targets":
            self._targets = None
        else:
            self._registration = None
            self._squad = None
            self._analysis = None
        return UploadReport(kind=kind, filename=filename, summary=summary, season=season, warnings=warnings)

    def _record_history(self, snapshot_id: int) -> None:
        a = self.analysis
        if a is None:
            return
        rows = []
        for p in a.players:
            dob = str(a.squad_df["DoB"].loc[p.idx]) if "DoB" in a.squad_df.columns else ""
            d, _ = parse_dob(dob)
            key = f"{p.name}|{d.isoformat()}" if d else p.name
            attrs = {c: float(a.squad_df[c].loc[p.idx]) for c in a.squad_df.columns
                     if c in ROLES[p.best_role_id].weights()} if p.best_role_id in ROLES else {}
            rows.append({
                "player_key": key, "name": p.name, "age": p.age, "season_label": self.season.label,
                "best_role": p.best_role_name, "best_score": p.best_score,
                "value_lo": p.value_lo, "value_hi": p.value_hi, "wage_k": p.wage_k, "attrs": attrs,
            })
        self.store.record_history(self.wid, snapshot_id, rows)

    # ---------------------------------------------------------------- pages
    def pitch_lineup(self) -> list[dict]:
        """Best XI with pitch coordinates and depth status, for the pitch graphic."""
        a = self.analysis
        if a is None:
            return []
        layout = pitch_layout(a.formation)
        lineup = []
        for i, b in enumerate(a.best_11):
            depth = a.slot_depth_status(i)
            lineup.append({
                **b, "slot_idx": i, "x": layout[i]["x"], "y": layout[i]["y"],
                "depth_status": depth["status"], "backup": depth["backup"], "n_backups": depth["n_backups"],
                "role_name": ROLES[b["role_id"]].name if b["role_id"] in ROLES else b["role_id"],
            })
        return lineup

    def dashboard(self) -> dict | None:
        a = self.analysis
        if a is None:
            return None
        shortlisted = self.store.shortlist_names(self.wid, "target")
        lineup = self.pitch_lineup()
        weak = a.weak_slots(3)
        upgrades_by_slot: dict[int, list[dict]] = {}
        if self.targets is not None and not self.targets.empty:
            ups = filter_formation_upgrades(
                self.targets, a.squad_df, a.formation, self.profile,
                min_margin=0.0, max_age=self.settings["max_age"], max_value=self.settings["max_value"],
                assumption=self.assumption, exclude_unscouted=self.settings["exclude_unscouted"], benchmarks=a.best_11,
            )
            for i in weak:
                pos = a.formation[i]["pos"]
                sub = ups[ups["Upgrade Position"] == pos].head(3)
                upgrades_by_slot[i] = [
                    {
                        "idx": int(ix), "name": r["Name"], "age": r.get("Age"), "club": r.get("Club", ""),
                        "score": r.get("Target Best Score"), "lo": r.get("Score Low"), "hi": r.get("Score High"),
                        "margin": r.get("Upgrade Margin"), "margin_lo": r.get("Margin Low"), "value": r.get("Transfer Value", ""),
                        "needs_scouting": bool(r.get("Needs Scouting", False)),
                        "shortlisted": r["Name"] in shortlisted,
                    }
                    for ix, r in sub.iterrows()
                ]
        sells = generate_sell_recommendations(a.squad_df, a.formation, self.profile, self.settings["max_youth_age"], self.settings["squad_size_limit"], analysis=a)
        sell_top = sells[sells["Recommendation"] == "Sell"].head(5).to_dict("records") if not sells.empty else []
        loan_top = sells[sells["Recommendation"] == "Loan"].head(5).to_dict("records") if not sells.empty else []
        youth = sorted((p for p in a.players if p.age <= self.settings["max_youth_age"]), key=lambda p: p.potential, reverse=True)[:5]
        reg = self.registration_result()
        return {
            "analysis": a,
            "lineup": lineup,
            "weak_slots": [{"slot_idx": i, "label": a.slot_label(i), "pos": a.formation[i]["pos"], "score": a.best_11[i]["score"],
                            "starter": a.best_11[i]["player_name"], "targets": upgrades_by_slot.get(i, [])} for i in weak],
            "depth_alerts": [{"slot_idx": i, "label": a.slot_label(i), **a.slot_depth_status(i)} for i in range(len(a.formation)) if a.slot_depth_status(i)["status"] != "ok"],
            "sell_top": sell_top,
            "loan_top": loan_top,
            "youth_top": [p.to_dict() for p in youth],
            "registration": reg,
            "total_quality": a.total_quality(),
            "average_quality": a.average_quality(),
            "squad_size": len(a.players),
            "wage_bill": round(sum(p.wage_k for p in a.players), 1),
            "unscouted_squad": sum(1 for p in a.players if p.needs_scouting),
        }

    def registration_result(self) -> RegistrationResult | None:
        a = self.analysis
        if a is None:
            return None
        return optimize_squad_registration(
            a.squad_df, a.formation, self.profile, max_squad=self.settings["max_squad"], min_hgn=self.settings["min_hgn"],
            u21_age=self.settings["u21_age"], season=self.season, analysis=a,
        )

    def squad_table(self) -> pd.DataFrame | None:
        a = self.analysis
        if a is None:
            return None
        rows = []
        for p in a.players:
            rows.append({
                "_idx": p.idx, "Name": p.name, "Age": p.age, "Position": p.position, "Personality": p.personality,
                "Best Role": p.best_role_name, "Score": p.best_score, "Low": p.best_score_lo, "High": p.best_score_hi,
                "Potential": p.potential, "Best Slot": a.formation[p.best_slot_idx]["pos"] if p.best_slot_idx is not None else "-",
                "Starter": "Yes" if p.is_starter else "", "Transfer Value": p.transfer_value, "Wage": p.wage,
                "Scouting %": p.scouting_pct, **{ROLES[r].name: s for r, s in p.role_scores.items()},
            })
        return pd.DataFrame(rows).sort_values("Score", ascending=False)

    def depth_chart(self) -> list[dict]:
        a = self.analysis
        return build_depth_chart(a) if a else []

    def upgrades(self, **overrides) -> pd.DataFrame | None:
        a = self.analysis
        if a is None or self.targets is None:
            return None
        s = {**self.settings, **overrides}
        ups = filter_formation_upgrades(
            self.targets, a.squad_df, a.formation, self.profile,
            min_margin=float(s["min_margin"]), max_age=int(s["max_age"]), max_value=str(s["max_value"]),
            position_mode=s["position_mode"], assumption=self.assumption, exclude_unscouted=bool(s["exclude_unscouted"]), benchmarks=a.best_11,
        )
        return ups.head(int(s["top_n"]))

    def youth(self) -> pd.DataFrame | None:
        a = self.analysis
        if a is None:
            return None
        benchmarks = {}
        for b in a.best_11:
            if b["player_idx"] >= 0 and (b["role_id"] not in benchmarks or b["score"] < benchmarks[b["role_id"]]["best_score"]):
                benchmarks[b["role_id"]] = {"best_score": b["score"], "best_player": b["player_name"]}
        return analyze_youth(a.squad_df, a.role_ids, self.profile, self.settings["max_youth_age"], benchmarks)

    def sell(self) -> pd.DataFrame | None:
        a = self.analysis
        if a is None:
            return None
        return generate_sell_recommendations(a.squad_df, a.formation, self.profile, self.settings["max_youth_age"], self.settings["squad_size_limit"], analysis=a)

    def strategy(self, **overrides) -> dict | None:
        a = self.analysis
        if a is None:
            return None
        s = {**self.settings, **overrides}
        return generate_strategy(
            a.squad_df, self.targets, a.formation, self.profile,
            transfer_budget=s["transfer_budget"], wage_budget=s["wage_budget"], seasons=int(s["seasons"]),
            max_transfers=int(s["max_transfers"]), locked_players=set(s["locked_players"] or []), u21_age=int(s["u21_age"]),
            assumption=self.assumption, min_gain=float(s["min_gain"]), min_gain_per_m=float(s["min_gain_per_m"]),
            exclude_unscouted=bool(s["exclude_unscouted"]), board_sales_percentage=float(s["board_sales_percentage"]), analysis=a,
        )

    def scenario(self, buys: list[str], sells: list[str], assumption: str | None = None, seasons: int | None = None) -> dict | None:
        a = self.analysis
        if a is None:
            return None
        asm = assumption if assumption in ASSUMPTIONS else self.assumption
        return evaluate_plan(a.squad_df, self.targets, a.formation, self.profile, buys, sells,
                             seasons=int(seasons or self.settings["seasons"]), assumption=asm,
                             analysis=a if asm == self.assumption else None)

    # --------------------------------------------------------------- player
    def player_detail(self, source: str, idx: int) -> dict | None:
        a = self.analysis
        df = (a.squad_df if a else None) if source == "squad" else self.targets
        if df is None or idx not in df.index:
            return None
        row = df.loc[idx]
        rec = a.by_idx.get(idx) if (source == "squad" and a) else None
        name = str(row["Name"])
        age_val = pd.to_numeric(row.get("Age"), errors="coerce")
        age = 0 if pd.isna(age_val) else int(age_val)
        personality = str(row.get("Personality", ""))
        position = str(row.get("Position", ""))
        role_rows = []
        for slot_idx, slot in enumerate(a.formation if a else self.slots):
            role = ROLES[slot["role"]]
            bd = score_breakdown(row, effective_role(slot["role"], self.profile))
            if a and source == "squad":
                fam = float(a.fam[slot_idx].loc[idx])
                incumbent = a.best_11[slot_idx]
            else:
                fam = position_familiarity(position, slot["role"], slot["pos"])
                incumbent = a.best_11[slot_idx] if a else None
            role_rows.append({
                "slot_idx": slot_idx, "pos": slot["pos"], "role_id": slot["role"], "role_name": role.name,
                "score": bd["score"], "lo": bd["lo"], "hi": bd["hi"], "familiarity": round(fam, 2),
                "effective": round(bd["score"] * fam, 1) if fam > 0 else None,
                "incumbent": incumbent["player_name"] if incumbent else None, "incumbent_score": incumbent["score"] if incumbent else None,
                "tiers": bd["tiers"],
            })
        role_rows.sort(key=lambda r: (r["effective"] or -1), reverse=True)
        best = role_rows[0] if role_rows else None
        signals = extract_signals(row)
        band = compute_potential_band(best["score"] if best else 0.0, age, personality, row)
        lo, mid, hi = project_band(best["score"] if best else 0.0, age, personality, position, 5, band["mid"], signals)
        attrs = {c: float(row[c]) for c in ROLES[best["role_id"]].weights() if c in row.index} if best else {}
        attr_ranges = {c: (float(row.get(f"{c}_lo", row[c])), float(row.get(f"{c}_hi", row[c]))) for c in attrs}
        vlo, vhi = parse_value_range(row.get("Transfer Value", ""))
        history = self.store.player_history(self.wid, self._player_key(row)) if source == "squad" else []
        shortlisted = name in self.store.shortlist_names(self.wid, "target" if source == "targets" else "squad")
        return {
            "source": source, "idx": int(idx), "name": name, "age": age, "position": position, "personality": personality,
            "club": str(row.get("Club", "")), "transfer_value": str(row.get("Transfer Value", "")), "wage": str(row.get("Wage", "")),
            "value_lo": vlo, "value_hi": vhi, "wage_k": parse_wage(row.get("Wage", "")),
            "scouting_pct": float(row.get("Scouting %", 100)), "needs_scouting": bool(row.get("Needs Scouting", False)),
            "roles": role_rows, "best": best, "attrs": attrs, "attr_ranges": attr_ranges,
            "signals": signals.to_dict(), "potential": band,
            "projection": {"seasons": list(range(len(mid))), "mid": mid, "lo": lo, "hi": hi},
            "record": rec.to_dict() if rec else None,
            "is_starter": bool(rec and rec.is_starter), "starter_slot": a.formation[rec.starter_slot_idx]["pos"] if rec and rec.is_starter else None,
            "history": history, "shortlisted": shortlisted,
            "hg_status": str(row.get("Home-Grown Status", "")), "statuses": str(row.get("Inf", "")),
        }

    @staticmethod
    def _player_key(row: pd.Series) -> str:
        d, _ = parse_dob(row.get("DoB", ""))
        return f"{row['Name']}|{d.isoformat()}" if d else str(row["Name"])

    def find_target_idx(self, name: str) -> int | None:
        if self.targets is None:
            return None
        m = self.targets.index[self.targets["Name"].astype(str) == name]
        return int(m[0]) if len(m) else None

    def find_squad_idx(self, name: str) -> int | None:
        if self.squad is None:
            return None
        m = self.squad.index[self.squad["Name"].astype(str) == name]
        return int(m[0]) if len(m) else None
