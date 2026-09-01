"""Shared squad model.

Every analysis view (Best 11, depth, sell, strategy, registration, dashboard)
used to rebuild its own idea of "who plays where" with slightly different
rules. :class:`SquadAnalysis` computes that once — slot-aware, with a globally
optimal starting XI — and the other modules read from it.
"""

from dataclasses import dataclass, field

import pandas as pd

from development import DevSignals, extract_signals
from money import parse_value_range, parse_wage
from positions import parse_slot_position
from profiles import Profile
from roles import ROLES
from scorer import familiarity_matrix, get_best_11, score_all_roles
from season import SeasonInfo, detect_season
from youth import compute_potential_band, get_personality_multiplier

DEPTH_THIN_GAP = 3.0  # backup this many points below the starter = thin cover


@dataclass
class PlayerRecord:
    idx: int
    name: str
    age: int
    position: str
    personality: str
    transfer_value: str
    wage: str
    value_lo: float
    value_hi: float
    wage_k: float
    best_role_id: str | None
    best_score: float
    best_score_lo: float
    best_score_hi: float
    best_slot_idx: int | None
    starter_slot_idx: int | None
    needs_scouting: bool
    scouting_pct: float
    potential: float
    potential_lo: float
    potential_hi: float
    growth_notes: list[str]
    signals: DevSignals
    pers_mult: float
    role_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_starter(self) -> bool:
        return self.starter_slot_idx is not None

    @property
    def best_role_name(self) -> str:
        return ROLES[self.best_role_id].name if self.best_role_id in ROLES else "-"

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "name": self.name,
            "age": self.age,
            "position": self.position,
            "personality": self.personality,
            "transfer_value": self.transfer_value,
            "wage": self.wage,
            "value_lo": self.value_lo,
            "value_hi": self.value_hi,
            "wage_k": self.wage_k,
            "best_role_id": self.best_role_id,
            "best_role": self.best_role_name,
            "best_score": self.best_score,
            "best_score_lo": self.best_score_lo,
            "best_score_hi": self.best_score_hi,
            "is_starter": self.is_starter,
            "needs_scouting": self.needs_scouting,
            "scouting_pct": self.scouting_pct,
            "potential": self.potential,
            "potential_lo": self.potential_lo,
            "potential_hi": self.potential_hi,
            "growth_notes": self.growth_notes,
        }


class SquadAnalysis:
    def __init__(
        self,
        squad_df: pd.DataFrame,
        formation: list[dict],
        profile: Profile,
        assumption: str = "mid",
        season: SeasonInfo | None = None,
    ):
        self.squad_df = squad_df
        self.formation = formation
        self.profile = profile
        self.assumption = assumption
        self.role_ids = list({s["role"] for s in formation})
        self.role_slot_count: dict[str, int] = {}
        for s in formation:
            self.role_slot_count[s["role"]] = self.role_slot_count.get(s["role"], 0) + 1
        self.season = season or detect_season(squad_df)

        self.scored = score_all_roles(squad_df, self.role_ids, profile, assumption, with_bounds=True)
        self.fam = familiarity_matrix(self.scored, formation)
        self.best_11 = get_best_11(squad_df, formation, profile, assumption, scored=self.scored)
        self.starters_by_idx: dict[int, int] = {
            b["player_idx"]: i for i, b in enumerate(self.best_11) if b["player_idx"] >= 0
        }
        self.depth: dict[int, list[dict]] = self._build_depth()
        self.players: list[PlayerRecord] = self._build_players()
        self.by_idx: dict[int, PlayerRecord] = {p.idx: p for p in self.players}

    # ------------------------------------------------------------------ #
    def slot_score(self, slot_idx: int, player_idx: int) -> float:
        role_id = self.formation[slot_idx]["role"]
        f = float(self.fam[slot_idx].loc[player_idx])
        if f <= 0 or role_id not in self.scored.columns:
            return 0.0
        return round(float(self.scored.at[player_idx, role_id]) * f, 1)

    def _build_depth(self) -> dict[int, list[dict]]:
        name_col = self.scored["Name"] if "Name" in self.scored.columns else pd.Series(["?"] * len(self.scored), index=self.scored.index)
        age_col = pd.to_numeric(self.scored.get("Age", pd.Series(99, index=self.scored.index)), errors="coerce").fillna(99)
        depth: dict[int, list[dict]] = {}
        for slot_idx, slot in enumerate(self.formation):
            rows = []
            f = self.fam[slot_idx]
            for idx in self.scored.index[f.values > 0]:
                rows.append({
                    "idx": int(idx),
                    "name": str(name_col.loc[idx]),
                    "age": int(age_col.loc[idx]),
                    "score": self.slot_score(slot_idx, idx),
                    "raw_score": round(float(self.scored.at[idx, slot["role"]]), 1),
                    "familiarity": float(f.loc[idx]),
                    "is_starter_here": self.starters_by_idx.get(int(idx)) == slot_idx,
                    "starts_elsewhere": int(idx) in self.starters_by_idx and self.starters_by_idx[int(idx)] != slot_idx,
                })
            rows.sort(key=lambda r: r["score"], reverse=True)
            depth[slot_idx] = rows
        return depth

    def _build_players(self) -> list[PlayerRecord]:
        s = self.scored
        n = len(s)
        idx_series = pd.Series(s.index, index=s.index)
        name_col = s["Name"] if "Name" in s.columns else idx_series.astype(str)
        age_col = pd.to_numeric(s.get("Age", pd.Series(99, index=s.index)), errors="coerce").fillna(99)
        pos_col = s["Position"].astype(str) if "Position" in s.columns else pd.Series([""] * n, index=s.index)
        pers_col = s["Personality"].astype(str) if "Personality" in s.columns else pd.Series([""] * n, index=s.index)
        val_col = s["Transfer Value"].astype(str) if "Transfer Value" in s.columns else pd.Series([""] * n, index=s.index)
        wage_col = s["Wage"].astype(str) if "Wage" in s.columns else pd.Series([""] * n, index=s.index)
        ns_col = s["Needs Scouting"] if "Needs Scouting" in s.columns else pd.Series(False, index=s.index)
        sp_col = s["Scouting %"] if "Scouting %" in s.columns else pd.Series(100.0, index=s.index)

        players: list[PlayerRecord] = []
        for idx in s.index:
            # Best slot = highest familiarity-weighted score across formation slots.
            best_slot, best_eff = None, -1.0
            for slot_idx in range(len(self.formation)):
                eff = self.slot_score(slot_idx, idx)
                if eff > best_eff and float(self.fam[slot_idx].loc[idx]) > 0:
                    best_slot, best_eff = slot_idx, eff
            if best_slot is None:
                # Cannot play any formation slot: still record best raw role.
                best_role = max(self.role_ids, key=lambda r: float(s.at[idx, r])) if self.role_ids else None
                best_score = float(s.at[idx, best_role]) if best_role else 0.0
            else:
                best_role = self.formation[best_slot]["role"]
                best_score = best_eff
            lo = float(s.at[idx, f"{best_role}_lo"]) if best_role and f"{best_role}_lo" in s.columns else best_score
            hi = float(s.at[idx, f"{best_role}_hi"]) if best_role and f"{best_role}_hi" in s.columns else best_score
            age = int(age_col.loc[idx])
            personality = str(pers_col.loc[idx])
            row = s.loc[idx]
            band = compute_potential_band(best_score, age, personality, row)
            vlo, vhi = parse_value_range(val_col.loc[idx])
            players.append(PlayerRecord(
                idx=int(idx),
                name=str(name_col.loc[idx]),
                age=age,
                position=str(pos_col.loc[idx]),
                personality=personality,
                transfer_value=str(val_col.loc[idx]),
                wage=str(wage_col.loc[idx]),
                value_lo=vlo,
                value_hi=vhi,
                wage_k=parse_wage(wage_col.loc[idx]),
                best_role_id=best_role,
                best_score=round(best_score, 1),
                best_score_lo=round(lo, 1),
                best_score_hi=round(hi, 1),
                best_slot_idx=best_slot,
                starter_slot_idx=self.starters_by_idx.get(int(idx)),
                needs_scouting=bool(ns_col.loc[idx]),
                scouting_pct=float(sp_col.loc[idx]),
                potential=band["mid"],
                potential_lo=band["lo"],
                potential_hi=band["hi"],
                growth_notes=band["notes"],
                signals=extract_signals(row),
                pers_mult=get_personality_multiplier(personality),
                role_scores={r: round(float(s.at[idx, r]), 1) for r in self.role_ids},
            ))
        return players

    # ------------------------------------------------------------------ #
    def starter_score_for_role(self, role_id: str) -> float | None:
        scores = [b["score"] for b in self.best_11 if b["role_id"] == role_id and b["player_idx"] >= 0]
        return min(scores) if scores else None

    def starter_score_for_slot(self, slot_idx: int) -> float:
        return float(self.best_11[slot_idx]["score"])

    def rank_at_slot(self, slot_idx: int, player_idx: int) -> int:
        for i, r in enumerate(self.depth.get(slot_idx, [])):
            if r["idx"] == player_idx:
                return i + 1
        return 999

    def backups_for_slot(self, slot_idx: int, exclude_starters: bool = True) -> list[dict]:
        rows = self.depth.get(slot_idx, [])
        out = []
        for r in rows:
            if r["is_starter_here"]:
                continue
            if exclude_starters and r["idx"] in self.starters_by_idx:
                continue
            out.append(r)
        return out

    def slot_depth_status(self, slot_idx: int) -> dict:
        """Depth diagnosis for a slot: critical / thin / ok plus best backup."""
        starter = self.best_11[slot_idx]
        backups = self.backups_for_slot(slot_idx)
        best_backup = backups[0] if backups else None
        if starter["player_idx"] < 0:
            status = "critical"
        elif best_backup is None:
            status = "critical"
        elif starter["score"] - best_backup["score"] > DEPTH_THIN_GAP:
            status = "thin"
        else:
            status = "ok"
        return {"status": status, "backup": best_backup, "n_backups": len(backups)}

    def total_quality(self) -> float:
        return round(sum(b["score"] for b in self.best_11), 1)

    def average_quality(self) -> float:
        n = len(self.best_11) or 1
        return round(self.total_quality() / n, 2)

    def weak_slots(self, n: int = 3) -> list[int]:
        order = sorted(range(len(self.best_11)), key=lambda i: self.best_11[i]["score"])
        return order[:n]

    def slot_label(self, slot_idx: int) -> str:
        slot = self.formation[slot_idx]
        return f"{slot['pos']} — {ROLES[slot['role']].name if slot['role'] in ROLES else slot['role']}"

    @staticmethod
    def slot_line(slot_pos: str) -> str:
        ptype, _ = parse_slot_position(slot_pos)
        return ptype or "?"
