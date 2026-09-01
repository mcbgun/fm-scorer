"""Sell / loan / promote recommendations built on the shared SquadAnalysis.

Every player is classified against the *slot* they are best at (side-aware),
using the globally-optimal Best XI as the starter set, so the depth chart,
Best 11 and these recommendations always agree.

Statuses: Starter, Backup, Surplus, Prospect, Stagnant, Declining.
"""

import math

import pandas as pd

from profiles import Profile
from roles import ROLES
from squad_model import PlayerRecord, SquadAnalysis

REC_ORDER = {"Sell": 0, "Loan": 1, "Promote": 2, "Keep — Backup": 3, "Keep — Prospect": 4, "Keep — Key": 5}


def build_depth_chart(analysis: SquadAnalysis) -> list[dict]:
    """Per-slot depth chart for display: [{slot, pos, role, players:[...]}]."""
    out = []
    for slot_idx, slot in enumerate(analysis.formation):
        status = analysis.slot_depth_status(slot_idx)
        out.append({
            "slot_idx": slot_idx,
            "pos": slot["pos"],
            "role_id": slot["role"],
            "role_name": ROLES[slot["role"]].name if slot["role"] in ROLES else slot["role"],
            "starter": analysis.best_11[slot_idx],
            "players": analysis.depth[slot_idx],
            "depth_status": status["status"],
            "n_backups": status["n_backups"],
        })
    return out


def _gap_to_starter(analysis: SquadAnalysis, p: PlayerRecord) -> tuple[float, str]:
    """Gap between the player and the starter at their best slot (positive = behind)."""
    if p.best_slot_idx is None:
        return 0.0, "no formation slot"
    starter = analysis.best_11[p.best_slot_idx]
    if starter["player_idx"] < 0:
        return -p.best_score, analysis.slot_label(p.best_slot_idx)
    return round(starter["score"] - p.best_score, 1), analysis.slot_label(p.best_slot_idx)


def classify_player(
    analysis: SquadAnalysis,
    p: PlayerRecord,
    in_squad_limit: bool,
    max_youth_age: int = 21,
    squad_size_limit: int = 25,
) -> tuple[str, str, str]:
    """Return (status, recommendation, reason) for a player."""
    gap, slot_label = _gap_to_starter(analysis, p)
    notes = f" ({'; '.join(p.growth_notes)})" if p.growth_notes else ""
    scout = " — attributes partly unknown, re-scout before deciding" if p.needs_scouting else ""

    if p.is_starter:
        slot = analysis.slot_label(p.starter_slot_idx)
        cover = analysis.slot_depth_status(p.starter_slot_idx)
        cover_txt = "no real cover behind" if cover["status"] == "critical" else ("thin cover behind" if cover["status"] == "thin" else "adequate cover")
        return ("Starter", "Keep — Key", f"Starts at {slot} ({p.best_score:.1f}); {cover_txt}{scout}")

    rank = analysis.rank_at_slot(p.best_slot_idx, p.idx) if p.best_slot_idx is not None else 999
    ahead = max(0, rank - 1)

    if p.age <= max_youth_age:
        if p.potential < 12.0 or (p.pers_mult < 0.95 and p.age >= 19):
            return ("Stagnant", "Sell", f"Potential only {p.potential:.1f} ({p.potential_lo:.1f}-{p.potential_hi:.1f}) with a {p.personality} personality — unlikely to reach the first team{notes}")
        if gap <= 0:
            return ("Prospect", "Promote", f"Already matches the starter at {slot_label} — promote and give minutes{notes}")
        if not in_squad_limit:
            if gap <= 5.0 and p.pers_mult >= 1.0:
                return ("Surplus", "Loan", f"{gap:.1f} behind the {slot_label} starter and outside the {squad_size_limit}-man squad — loan for senior minutes{notes}")
            return ("Surplus", "Sell", f"Outside the {squad_size_limit}-man squad, {gap:.1f} behind at {slot_label}, potential {p.potential:.1f} doesn't justify a place{notes}")
        if gap <= 2.0:
            return ("Prospect", "Keep — Prospect", f"{gap:.1f} behind the {slot_label} starter, potential {p.potential:.1f} ({p.potential_lo:.1f}-{p.potential_hi:.1f}) — first-team ready soon{notes}")
        if gap <= 5.0:
            return ("Prospect", "Loan", f"{gap:.1f} behind at {slot_label}; needs regular football to close the gap (potential {p.potential:.1f}){notes}")
        return ("Prospect", "Keep — Prospect", f"Long-term project: {gap:.1f} behind at {slot_label}, potential {p.potential:.1f} ({p.potential_lo:.1f}-{p.potential_hi:.1f}){notes}")

    if not in_squad_limit:
        if p.age >= 28:
            return ("Surplus", "Sell", f"Age {p.age}, outside the {squad_size_limit}-man squad, {gap:.1f} behind at {slot_label} — value will only fall{scout}")
        return ("Surplus", "Sell", f"Outside the {squad_size_limit}-man squad; {ahead} player(s) ahead at {slot_label}{scout}")

    if p.age >= 29 and gap > 1.5:
        return ("Declining", "Sell", f"Age {p.age}, {gap:.1f} behind at {slot_label}; sell before value drops and free a squad place{scout}")
    if p.age >= 30 and gap > 0.5:
        return ("Declining", "Sell", f"Age {p.age}, {gap:.1f} behind at {slot_label} — sell now while there is still resale value{scout}")
    if p.age >= 28:
        return ("Backup", "Keep — Backup", f"Experienced cover at {slot_label}, {gap:.1f} behind the starter{scout}")
    return ("Backup", "Keep — Backup", f"Depth at {slot_label}, {gap:.1f} behind the starter{scout}")


def compute_sell_priority(analysis: SquadAnalysis, p: PlayerRecord) -> float:
    """Higher = sell first. Combines surplus gap, age, value, congestion,
    personality (youth) and potential (youth)."""
    gap, _ = _gap_to_starter(analysis, p)
    surplus_factor = max(0.0, gap) * 2.0
    age_factor = max(0, p.age - 24) * 0.5
    val_factor = math.log10(p.value_lo + 1) * 0.5 if p.value_lo > 0 else 0.0
    rank = analysis.rank_at_slot(p.best_slot_idx, p.idx) if p.best_slot_idx is not None else 1
    congestion_factor = max(0, rank - 1) * 0.3
    pers_penalty = (1.0 - p.pers_mult) * 3.0 if p.age <= 21 and p.pers_mult < 1.0 else 0.0
    potential_offset = -p.potential * 0.3 if p.age <= 21 else 0.0
    return round(surplus_factor + age_factor + val_factor + congestion_factor + pers_penalty + potential_offset, 2)


def youth_prospects(analysis: SquadAnalysis, max_youth_age: int = 21) -> set[int]:
    """Young players good enough to be exempt from the squad-size count."""
    out = set()
    for p in analysis.players:
        if p.age > max_youth_age:
            continue
        gap, _ = _gap_to_starter(analysis, p)
        if p.pers_mult >= 1.0 and p.potential >= 13.0 and (gap <= 3.0 or (p.potential >= 14.0 and p.age <= 17)):
            out.add(p.idx)
    return out


def generate_sell_recommendations(
    squad_df: pd.DataFrame,
    formation_slots: list[dict],
    profile: Profile,
    max_youth_age: int = 21,
    squad_size_limit: int = 25,
    analysis: SquadAnalysis | None = None,
) -> pd.DataFrame:
    """Sell/loan/keep recommendations for every squad player."""
    if not formation_slots:
        formation_slots = [{"pos": "GK", "role": "gkd"}]
    if analysis is None:
        analysis = SquadAnalysis(squad_df, formation_slots, profile)

    prospects = youth_prospects(analysis, max_youth_age)
    non_prospects = sorted((p for p in analysis.players if p.idx not in prospects), key=lambda p: p.best_score, reverse=True)
    squad_ids = {p.idx for p in non_prospects[:squad_size_limit]}

    rows = []
    for p in analysis.players:
        status, rec, reason = classify_player(analysis, p, p.idx in squad_ids, max_youth_age, squad_size_limit)
        slot_idx = p.starter_slot_idx if p.is_starter else p.best_slot_idx
        rows.append({
            "Name": p.name,
            "Age": p.age,
            "Position": p.position,
            "Personality": p.personality,
            "Transfer Value": p.transfer_value,
            "Wage": p.wage,
            "Best Slot": analysis.formation[slot_idx]["pos"] if slot_idx is not None else "-",
            "Best Formation Role": p.best_role_name,
            "Role Score": p.best_score,
            "Score Low": p.best_score_lo,
            "Score High": p.best_score_hi,
            "Rank at Slot": analysis.rank_at_slot(slot_idx, p.idx) if slot_idx is not None else 999,
            "Potential": p.potential,
            "Status": status,
            "Recommendation": rec,
            "Sell Priority": compute_sell_priority(analysis, p),
            "Reason": reason,
            "Needs Scouting": p.needs_scouting,
            "_idx": p.idx,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_rec_order"] = result["Recommendation"].map(lambda r: REC_ORDER.get(r, 9))
    result = result.sort_values(["_rec_order", "Sell Priority"], ascending=[True, False]).drop(columns=["_rec_order"])
    return result.reset_index(drop=True)
