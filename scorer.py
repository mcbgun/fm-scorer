"""Scoring engine: role scores (with confidence bounds), slot-aware Best 11 and
upgrade detection.

Assumptions
-----------
Attributes exported as ranges ("12-16") carry ``_lo`` / ``_hi`` columns (see
``parser.add_confidence_columns``). Every scoring function accepts an
``assumption``:

- ``"mid"``  - midpoint of each range (headline score)
- ``"low"``  - conservative (every unknown resolves to its lower bound)
- ``"high"`` - optimistic
"""

import pandas as pd

from assignment import INCOMPATIBLE, assign_slots
from money import parse_value_high
from positions import position_familiarity
from profiles import Profile, apply_profile
from roles import ROLES, RoleDef

ASSUMPTIONS = ("low", "mid", "high")
FAMILIARITY_SUFFIX = "__fam"


def _attr_frame(df: pd.DataFrame, attrs, assumption: str) -> list[pd.Series]:
    suffix = {"low": "_lo", "high": "_hi"}.get(assumption, "")
    out = []
    for a in attrs:
        col = f"{a}{suffix}" if suffix and f"{a}{suffix}" in df.columns else a
        out.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index))
    return out


def score_role(df: pd.DataFrame, role: RoleDef, assumption: str = "mid") -> pd.Series:
    """Compute a single role score for all players in the dataframe."""
    denom = role.denominator
    if denom == 0 or len(df) == 0:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float)
    key_sum = sum(_attr_frame(df, role.key, assumption), pd.Series(0.0, index=df.index))
    green_sum = sum(_attr_frame(df, role.green, assumption), pd.Series(0.0, index=df.index))
    blue_sum = sum(_attr_frame(df, role.blue, assumption), pd.Series(0.0, index=df.index))
    score = (key_sum * 5 + green_sum * 3 + blue_sum * 1) / denom
    return score.round(2)


def score_all_roles(
    df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    assumption: str = "mid",
    with_bounds: bool = False,
) -> pd.DataFrame:
    """Score all selected roles for all players.

    Adds one column per role id. With ``with_bounds`` also adds ``<role>_lo`` /
    ``<role>_hi`` columns (conservative / optimistic scores).
    """
    result = df.copy()
    new_cols: dict[str, pd.Series] = {}
    for role_id in selected_role_ids:
        if role_id not in ROLES:
            continue
        role = apply_profile(ROLES[role_id], profile)
        new_cols[role_id] = score_role(df, role, assumption)
        if with_bounds:
            new_cols[f"{role_id}_lo"] = score_role(df, role, "low")
            new_cols[f"{role_id}_hi"] = score_role(df, role, "high")
    if new_cols:
        result = result.drop(columns=[c for c in new_cols if c in result.columns])
        result = pd.concat([result, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return result


def score_breakdown(row: pd.Series, role: RoleDef) -> dict:
    """Per-tier contribution for one player and one (profile-applied) role."""
    tiers = {}
    for tier, attrs, w in (("key", role.key, 5), ("green", role.green, 3), ("blue", role.blue, 1)):
        items = []
        for a in attrs:
            val = float(row.get(a, 0) or 0)
            lo = float(row.get(f"{a}_lo", val) or 0)
            hi = float(row.get(f"{a}_hi", val) or 0)
            items.append({"attr": a, "value": val, "lo": lo, "hi": hi, "weight": w})
        tiers[tier] = {"weight": w, "attrs": items, "avg": round(sum(i["value"] for i in items) / len(items), 1) if items else 0.0}
    denom = role.denominator or 1
    total = sum(i["value"] * i["weight"] for t in tiers.values() for i in t["attrs"]) / denom
    lo = sum(i["lo"] * i["weight"] for t in tiers.values() for i in t["attrs"]) / denom
    hi = sum(i["hi"] * i["weight"] for t in tiers.values() for i in t["attrs"]) / denom
    return {"role_id": role.id, "role_name": role.name, "score": round(total, 1), "lo": round(lo, 1), "hi": round(hi, 1), "tiers": tiers}


def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spd, Work, SetP derived columns (matching Squirrel's app)."""
    df = df.copy()
    df["Spd"] = ((df["Pac"] + df["Acc"]) / 2).round(1)
    df["Work"] = ((df["Wor"] + df["Sta"]) / 2).round(1)
    df["SetP"] = ((df["Jum"] + df["Bra"]) / 2).round(1)
    return df


def get_best_role_per_player(df: pd.DataFrame, selected_role_ids: list[str], profile: Profile) -> pd.DataFrame:
    """Add 'Highest Role Score' and 'Resulting Role' columns."""
    scored = score_all_roles(df, selected_role_ids, profile, with_bounds=True)
    valid = [r for r in selected_role_ids if r in scored.columns]
    if not valid:
        scored["Highest Role Score"] = None
        scored["Resulting Role"] = ""
        return scored
    role_cols = scored[valid]
    scored["Highest Role Score"] = role_cols.max(axis=1).round(1)
    best_idx = role_cols.idxmax(axis=1)
    scored["Resulting Role"] = best_idx.map(lambda rid: ROLES[rid].name)
    scored["Score Low"] = pd.Series([scored.at[i, f"{r}_lo"] for i, r in best_idx.items()], index=scored.index).round(1)
    scored["Score High"] = pd.Series([scored.at[i, f"{r}_hi"] for i, r in best_idx.items()], index=scored.index).round(1)
    return scored


# --------------------------------------------------------------------------- #
# Slot-aware assignment
# --------------------------------------------------------------------------- #


def familiarity_matrix(df: pd.DataFrame, slots: list[dict], wrong_flank: float | None = None) -> dict[int, pd.Series]:
    """{slot_idx: Series(familiarity per player)} — vectorised over unique
    position strings so 16k targets x 11 slots stays fast."""
    pos_col = df["Position"].astype(str) if "Position" in df.columns else pd.Series([""] * len(df), index=df.index)
    uniques = pos_col.unique()
    out: dict[int, pd.Series] = {}
    kwargs = {} if wrong_flank is None else {"wrong_flank": wrong_flank}
    for i, slot in enumerate(slots):
        lookup = {p: position_familiarity(p, slot["role"], slot["pos"], **kwargs) for p in uniques}
        out[i] = pos_col.map(lookup).astype(float)
    return out


def get_best_11(
    squad_df: pd.DataFrame,
    formation: list[dict],
    profile: Profile,
    assumption: str = "mid",
    exclude_idx: set | None = None,
    wrong_flank: float | None = None,
    scored: pd.DataFrame | None = None,
) -> list[dict]:
    """Assign squad players to formation slots with a globally optimal
    (Hungarian) matching, respecting the *side* of each slot.

    Slot score = role score x positional familiarity. Returns one dict per slot
    (in formation order); unfilled slots have ``player_name="(no one)"``.
    """
    unique_role_ids = list({slot["role"] for slot in formation})
    if scored is None:
        scored = score_all_roles(squad_df, unique_role_ids, profile, assumption, with_bounds=True)
    if exclude_idx:
        scored = scored.loc[[i for i in scored.index if i not in exclude_idx]]
    name_col = scored["Name"] if "Name" in scored.columns else pd.Series(["?"] * len(scored), index=scored.index)
    pos_col = scored["Position"].astype(str) if "Position" in scored.columns else pd.Series([""] * len(scored), index=scored.index)
    fam = familiarity_matrix(scored, formation, wrong_flank)

    role_scores = {rid: scored[rid] for rid in unique_role_ids if rid in scored.columns}

    def slot_score(slot_idx, player_idx):
        role_id = formation[slot_idx]["role"]
        if role_id not in role_scores:
            return INCOMPATIBLE
        f = float(fam[slot_idx].loc[player_idx])
        if f <= 0:
            return INCOMPATIBLE
        return float(role_scores[role_id].loc[player_idx]) * f

    assignment = assign_slots(list(range(len(formation))), list(scored.index), slot_score)

    results = []
    for slot_idx, slot in enumerate(formation):
        role_id = slot["role"]
        player_idx, eff = assignment[slot_idx]
        base = {"pos": slot["pos"], "role_id": role_id, "role_name": ROLES[role_id].name if role_id in ROLES else role_id}
        if player_idx is None:
            results.append({**base, "player_idx": -1, "player_name": "(no one)", "score": 0.0, "raw_score": 0.0,
                            "score_lo": 0.0, "score_hi": 0.0, "familiarity": 0.0, "position": "", "needs_scouting": False})
            continue
        raw = float(role_scores[role_id].loc[player_idx])
        lo_col, hi_col = f"{role_id}_lo", f"{role_id}_hi"
        lo = float(scored.at[player_idx, lo_col]) if lo_col in scored.columns else raw
        hi = float(scored.at[player_idx, hi_col]) if hi_col in scored.columns else raw
        results.append({
            **base,
            "player_idx": int(player_idx),
            "player_name": str(name_col.loc[player_idx]),
            "score": round(eff, 1),
            "raw_score": round(raw, 1),
            "score_lo": round(lo, 1),
            "score_hi": round(hi, 1),
            "familiarity": float(fam[slot_idx].loc[player_idx]),
            "position": str(pos_col.loc[player_idx]),
            "needs_scouting": bool(scored.at[player_idx, "Needs Scouting"]) if "Needs Scouting" in scored.columns else False,
        })
    return results


def get_squad_benchmarks(
    squad_df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    one_player_per_role: bool = True,
    assumption: str = "mid",
) -> dict[str, dict]:
    """Benchmark score per role from the current squad.

    With ``one_player_per_role`` the squad is matched to roles with an optimal
    assignment (each player counts for one role only); position compatibility
    is always respected.
    """
    scored = score_all_roles(squad_df, selected_role_ids, profile, assumption)
    valid = [r for r in selected_role_ids if r in scored.columns]
    name_col = scored["Name"] if "Name" in scored.columns else pd.Series(["?"] * len(scored), index=scored.index)
    pseudo_slots = [{"pos": "", "role": r} for r in valid]
    fam = familiarity_matrix(scored, pseudo_slots)

    benchmarks: dict[str, dict] = {}
    if not one_player_per_role:
        for i, role_id in enumerate(valid):
            eligible = scored[role_id].where(fam[i] > 0, -1.0).sort_values(ascending=False)
            if eligible.empty or eligible.iloc[0] < 0:
                benchmarks[role_id] = {"best_score": 0.0, "best_player": "(no one)", "second_score": 0.0}
                continue
            second = float(eligible.iloc[1]) if len(eligible) > 1 and eligible.iloc[1] >= 0 else 0.0
            benchmarks[role_id] = {"best_score": float(eligible.iloc[0]), "best_player": str(name_col.loc[eligible.index[0]]), "second_score": second}
        return benchmarks

    def slot_score(slot_idx, player_idx):
        f = float(fam[slot_idx].loc[player_idx])
        return INCOMPATIBLE if f <= 0 else float(scored.at[player_idx, valid[slot_idx]])

    assignment = assign_slots(list(range(len(valid))), list(scored.index), slot_score)
    for i, role_id in enumerate(valid):
        player_idx, score = assignment[i]
        if player_idx is None:
            benchmarks[role_id] = {"best_score": 0.0, "best_player": "(unassigned)", "second_score": 0.0}
            continue
        others = scored.loc[(scored.index != player_idx) & (fam[i] > 0), role_id]
        benchmarks[role_id] = {
            "best_score": round(float(score), 1),
            "best_player": str(name_col.loc[player_idx]),
            "second_score": round(float(others.max()), 1) if len(others) else 0.0,
        }
    return benchmarks


def get_formation_benchmarks(squad_df: pd.DataFrame, formation: list[dict], profile: Profile, assumption: str = "mid") -> list[dict]:
    return get_best_11(squad_df, formation, profile, assumption)


def _apply_common_filters(scored: pd.DataFrame, max_age: int, max_value: str, exclude_unscouted: bool) -> pd.DataFrame:
    if "Age" in scored.columns:
        scored = scored.assign(Age=pd.to_numeric(scored["Age"], errors="coerce").fillna(99))
        scored = scored[scored["Age"] <= max_age]
    if max_value and "Transfer Value" in scored.columns:
        max_val_num = parse_value_high(max_value)
        if max_val_num > 0:
            vals = scored["Transfer Value"].apply(parse_value_high)
            scored = scored[vals <= max_val_num]
    if exclude_unscouted and "Needs Scouting" in scored.columns:
        scored = scored[~scored["Needs Scouting"].astype(bool)]
    return scored


def filter_upgrades(
    targets_df: pd.DataFrame,
    squad_df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    min_margin: float = 0.0,
    max_age: int = 99,
    max_value: str = "",
    require_strict_upgrade: bool = True,
    position_mode: str = "can_play",
    one_player_per_role: bool = True,
    assumption: str = "mid",
    exclude_unscouted: bool = False,
) -> pd.DataFrame:
    """Role-based (no formation) upgrade search."""
    benchmarks = get_squad_benchmarks(squad_df, selected_role_ids, profile, one_player_per_role)
    scored = score_all_roles(targets_df, selected_role_ids, profile, assumption, with_bounds=True)
    valid = [r for r in selected_role_ids if r in scored.columns]
    if not valid:
        return scored

    fam = familiarity_matrix(scored, [{"pos": "", "role": r} for r in valid])
    eff = pd.DataFrame({r: scored[r] for r in valid})
    for i, r in enumerate(valid):
        if position_mode == "can_play":
            eff[r] = eff[r].where(fam[i] > 0, -999.0)
        elif position_mode == "cannot_play":
            eff[r] = eff[r].where(fam[i] <= 0, -999.0)

    scored["Target Best Score"] = eff.max(axis=1).round(1)
    best_role_id = eff.idxmax(axis=1)
    scored["Target Best Role"] = best_role_id.map(lambda rid: ROLES[rid].name)
    scored["Target Best Role ID"] = best_role_id
    scored["Score Low"] = pd.Series([scored.at[i, f"{r}_lo"] for i, r in best_role_id.items()], index=scored.index).round(1)
    scored["Score High"] = pd.Series([scored.at[i, f"{r}_hi"] for i, r in best_role_id.items()], index=scored.index).round(1)
    scored["Squad Best Score"] = best_role_id.map(lambda rid: benchmarks.get(rid, {}).get("best_score", 0.0))
    scored["Squad Best Player"] = best_role_id.map(lambda rid: benchmarks.get(rid, {}).get("best_player", ""))
    scored["Upgrade Margin"] = (scored["Target Best Score"] - scored["Squad Best Score"]).round(1)
    scored["Margin Low"] = (scored["Score Low"] - scored["Squad Best Score"]).round(1)

    if require_strict_upgrade:
        mask = (scored["Upgrade Margin"] > min_margin) & (scored["Target Best Score"] > 0)
    else:
        mask = (scored["Upgrade Margin"] >= min_margin) & (scored["Target Best Score"] > 0)
    scored = _apply_common_filters(scored[mask], max_age, max_value, exclude_unscouted)
    return scored.sort_values("Upgrade Margin", ascending=False)


def filter_formation_upgrades(
    targets_df: pd.DataFrame,
    squad_df: pd.DataFrame,
    formation: list[dict],
    profile: Profile,
    min_margin: float = 0.0,
    max_age: int = 99,
    max_value: str = "",
    position_mode: str = "can_play",
    assumption: str = "mid",
    exclude_unscouted: bool = False,
    benchmarks: list[dict] | None = None,
) -> pd.DataFrame:
    """Find transfer targets who upgrade any formation slot (side-aware).

    For each target the slot with the largest ``score x familiarity -
    incumbent`` margin is reported, together with the conservative margin
    (``Margin Low``) so partially-scouted players are not over-sold.
    """
    if benchmarks is None:
        benchmarks = get_best_11(squad_df, formation, profile, assumption)
    unique_role_ids = list({slot["role"] for slot in formation})
    scored = score_all_roles(targets_df, unique_role_ids, profile, assumption, with_bounds=True)
    if not unique_role_ids or scored.empty:
        return scored

    fam = familiarity_matrix(scored, formation)
    n = len(scored)
    best_score = pd.Series(-999.0, index=scored.index)
    best_margin = pd.Series(-999.0, index=scored.index)
    best_lo = pd.Series(0.0, index=scored.index)
    best_hi = pd.Series(0.0, index=scored.index)
    best_slot = pd.Series([""] * n, index=scored.index)
    best_role = pd.Series([""] * n, index=scored.index)
    best_pos = pd.Series([""] * n, index=scored.index)
    beaten = pd.Series([""] * n, index=scored.index)
    best_fam = pd.Series(0.0, index=scored.index)

    for slot_idx, bm in enumerate(benchmarks):
        role_id = bm["role_id"]
        if role_id not in scored.columns:
            continue
        f = fam[slot_idx]
        if position_mode == "can_play":
            eligible = f > 0
        elif position_mode == "cannot_play":
            eligible = f <= 0
            f = pd.Series(1.0, index=scored.index)
        else:
            eligible = pd.Series(True, index=scored.index)
            f = f.where(f > 0, 1.0)
        eff = scored[role_id] * f
        margins = (eff - bm["score"]).where(eligible, -999.0)
        better = margins > best_margin
        best_score = best_score.where(~better, eff.round(1))
        best_margin = best_margin.where(~better, margins)
        best_lo = best_lo.where(~better, (scored[f"{role_id}_lo"] * f).round(1))
        best_hi = best_hi.where(~better, (scored[f"{role_id}_hi"] * f).round(1))
        best_fam = best_fam.where(~better, f)
        best_slot = best_slot.where(~better, f"{bm['pos']} ({role_id})")
        best_role = best_role.where(~better, ROLES[role_id].name)
        best_pos = best_pos.where(~better, bm["pos"])
        beaten = beaten.where(~better, bm["player_name"])

    bm_by_pos = {bm["pos"]: bm["score"] for bm in benchmarks}
    scored["Target Best Score"] = best_score.round(1)
    scored["Score Low"] = best_lo
    scored["Score High"] = best_hi
    scored["Upgrade Margin"] = best_margin.round(1)
    scored["Margin Low"] = (best_lo - best_pos.map(bm_by_pos).fillna(0.0)).round(1)
    scored["Familiarity"] = best_fam.round(2)
    scored["Upgrade Slot"] = best_slot
    scored["Upgrade Position"] = best_pos
    scored["Upgrade Role"] = best_role
    scored["Squad Player Beaten"] = beaten

    mask = (scored["Upgrade Margin"] > min_margin) & (scored["Target Best Score"] > 0)
    scored = _apply_common_filters(scored[mask], max_age, max_value, exclude_unscouted)
    return scored.sort_values("Upgrade Margin", ascending=False)
