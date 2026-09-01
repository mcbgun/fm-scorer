"""Scoring engine: computes role scores from player attribute data."""

import re

import pandas as pd

from roles import ROLES, RoleDef
from profiles import PROFILES, Profile, apply_profile
from positions import player_can_play_role


def score_role(df: pd.DataFrame, role: RoleDef) -> pd.Series:
    """Compute a single role score for all players in the dataframe.

    Args:
        df: DataFrame with player attribute columns (Acc, Pac, Sta, etc.)
        role: RoleDef with key/green/blue attribute lists

    Returns:
        Series of rounded role scores
    """
    denom = role.denominator
    if denom == 0:
        return pd.Series([0.0] * len(df), index=df.index)
    key_sum = sum(df[attr] for attr in role.key)
    green_sum = sum(df[attr] for attr in role.green)
    blue_sum = sum(df[attr] for attr in role.blue)
    score = (key_sum * 5 + green_sum * 3 + blue_sum * 1) / denom
    return score.round(1)


def score_all_roles(
    df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
) -> pd.DataFrame:
    """Score all selected roles for all players.

    Args:
        df: DataFrame with player attributes + info columns
        selected_role_ids: list of role IDs to score
        profile: weighting profile to apply

    Returns:
        DataFrame with original info columns + one column per selected role
    """
    result = df.copy()

    for role_id in selected_role_ids:
        base_role = ROLES[role_id]
        role = apply_profile(base_role, profile)
        result[role_id] = score_role(df, role)

    return result


def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spd, Work, SetP derived columns (matching Squirrel's app)."""
    df = df.copy()
    df["Spd"] = ((df["Pac"] + df["Acc"]) / 2).round(1)
    df["Work"] = ((df["Wor"] + df["Sta"]) / 2).round(1)
    df["SetP"] = ((df["Jum"] + df["Bra"]) / 2).round(1)
    return df


def get_best_role_per_player(
    df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
) -> pd.DataFrame:
    """Add 'Highest Role Score' and 'Resulting Role' columns."""
    scored = score_all_roles(df, selected_role_ids, profile)
    if not selected_role_ids:
        scored["Highest Role Score"] = None
        scored["Resulting Role"] = ""
        return scored

    role_cols = scored[selected_role_ids]
    scored["Highest Role Score"] = role_cols.max(axis=1).round(1)
    best_idx = role_cols.idxmax(axis=1)
    scored["Resulting Role"] = best_idx.map(lambda rid: ROLES[rid].name)

    return scored


def get_squad_benchmarks(
    squad_df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    one_player_per_role: bool = True,
) -> dict[str, dict]:
    """Compute benchmark scores from the current squad for each role.

    When one_player_per_role is True, each squad player is assigned to only
    their single best role using a greedy maximum-weight assignment. This
    prevents one player from being the benchmark for multiple roles.

    Args:
        squad_df: DataFrame of squad players with attributes
        selected_role_ids: roles to benchmark
        profile: weighting profile to apply
        one_player_per_role: if True, each player only counts for one role

    Returns:
        {role_id: {"best_score": float, "best_player": str, "second_score": float}}
    """
    scored = score_all_roles(squad_df, selected_role_ids, profile)
    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))

    if not one_player_per_role:
        # Original behavior: each role independently finds its best player
        benchmarks = {}
        for role_id in selected_role_ids:
            if role_id not in scored.columns:
                continue
            role_scores = scored[role_id]
            sorted_idx = role_scores.sort_values(ascending=False).index
            best_score = float(role_scores.loc[sorted_idx[0]])
            best_player = str(name_col.loc[sorted_idx[0]])
            second_score = float(role_scores.loc[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            benchmarks[role_id] = {
                "best_score": best_score,
                "best_player": best_player,
                "second_score": second_score,
            }
        return benchmarks

    # Greedy one-player-per-role assignment:
    # 1. Build all (player_idx, role_id, score) tuples
    # 2. Sort by score descending
    # 3. Assign greedily — skip if player or role already taken
    assignments: list[tuple[int, str, float]] = []
    for role_id in selected_role_ids:
        if role_id not in scored.columns:
            continue
        for idx in scored.index:
            assignments.append((idx, role_id, float(scored.at[idx, role_id])))

    assignments.sort(key=lambda x: x[2], reverse=True)

    assigned_players: set[int] = set()
    assigned_roles: set[str] = set()
    role_assignment: dict[str, tuple[int, float]] = {}

    for player_idx, role_id, score in assignments:
        if player_idx in assigned_players or role_id in assigned_roles:
            continue
        role_assignment[role_id] = (player_idx, score)
        assigned_players.add(player_idx)
        assigned_roles.add(role_id)
        if len(assigned_roles) == len(selected_role_ids):
            break

    # Build benchmarks from assignments, find second-best (excluding assigned player)
    benchmarks = {}
    for role_id in selected_role_ids:
        if role_id not in scored.columns:
            continue
        if role_id in role_assignment:
            player_idx, best_score = role_assignment[role_id]
            best_player = str(name_col.loc[player_idx])
            # Second best: highest score among non-assigned players
            other_scores = scored.loc[scored.index != player_idx, role_id]
            second_score = float(other_scores.max()) if len(other_scores) > 0 else 0.0
        else:
            # No player assigned to this role (not enough squad players)
            best_score = 0.0
            best_player = "(unassigned)"
            second_score = 0.0
        benchmarks[role_id] = {
            "best_score": best_score,
            "best_player": best_player,
            "second_score": second_score,
        }

    return benchmarks


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
) -> pd.DataFrame:
    """Filter transfer targets to show only upgrades over the current squad.

    For each target player, compares their best role score against the squad's
    best score for that same role. Only includes players who improve on the
    squad benchmark by at least min_margin.

    Args:
        targets_df: DataFrame of transfer target players
        squad_df: DataFrame of current squad players
        selected_role_ids: roles to evaluate
        profile: weighting profile to apply
        min_margin: minimum score improvement over squad best (e.g. 0.5)
        max_age: exclude players older than this
        max_value: exclude players with transfer value above this (string like "50M")
        require_strict_upgrade: if True, target must beat squad best (not just equal)
        position_mode: "any" (no filter), "can_play" (only players who can play
            the role's position), "cannot_play" (only players who'd need training)
        one_player_per_role: if True, squad benchmarks assign each player to one role

    Returns:
        DataFrame of upgrade targets with upgrade columns added
    """
    benchmarks = get_squad_benchmarks(squad_df, selected_role_ids, profile, one_player_per_role)
    scored = score_all_roles(targets_df, selected_role_ids, profile)

    if not selected_role_ids:
        return scored

    # Position filtering: mask each player's role scores based on position_mode
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    if position_mode == "can_play":
        for role_id in selected_role_ids:
            mask = pos_col.apply(lambda p: player_can_play_role(str(p), role_id))
            scored.loc[~mask, role_id] = -999  # disqualify from being best role
    elif position_mode == "cannot_play":
        for role_id in selected_role_ids:
            mask = pos_col.apply(lambda p: player_can_play_role(str(p), role_id))
            scored.loc[mask, role_id] = -999  # disqualify from being best role

    role_cols = scored[selected_role_ids]
    # Find each target's best eligible role
    scored["Target Best Score"] = role_cols.max(axis=1).round(1)
    best_role_id = role_cols.idxmax(axis=1)
    scored["Target Best Role"] = best_role_id.map(lambda rid: ROLES[rid].name)
    scored["Target Best Role ID"] = best_role_id

    # Squad benchmark for each player's best role
    scored["Squad Best Score"] = best_role_id.map(
        lambda rid: benchmarks.get(rid, {}).get("best_score", 0.0)
    )
    scored["Squad Best Player"] = best_role_id.map(
        lambda rid: benchmarks.get(rid, {}).get("best_player", "")
    )
    scored["Upgrade Margin"] = (scored["Target Best Score"] - scored["Squad Best Score"]).round(1)

    # Filter: must be an upgrade (and not disqualified by position)
    if require_strict_upgrade:
        mask = (scored["Upgrade Margin"] > min_margin) & (scored["Target Best Score"] > 0)
    else:
        mask = (scored["Upgrade Margin"] >= min_margin) & (scored["Target Best Score"] > 0)
    scored = scored[mask]

    # Filter: age
    if "Age" in scored.columns:
        scored["Age"] = pd.to_numeric(scored["Age"], errors="coerce").fillna(99)
        scored = scored[scored["Age"] <= max_age]

    # Filter: transfer value
    if max_value and "Transfer Value" in scored.columns:
        max_val_num = _parse_value_string(max_value)
        if max_val_num > 0:
            scored["_value_num"] = scored["Transfer Value"].apply(_parse_value_string)
            scored = scored[scored["_value_num"] <= max_val_num]
            scored = scored.drop(columns=["_value_num"])

    scored = scored.sort_values("Upgrade Margin", ascending=False)
    return scored


def _parse_value_string(val: str) -> float:
    """Parse an FM24 transfer value string into a numeric (in millions).

    Handles formats like "£34M - £55M", "£7.5K", "£1.2M", "-", "".
    Returns the upper bound in millions (e.g. "£34M - £55M" -> 55.0).
    """
    if not val or pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s == "-" or s == "":
        return 0.0

    matches = re.findall(r"[\$£€]?([\d.,]+)\s*([KMB]?)", s)
    values = []
    for num_str, unit in matches:
        try:
            num = float(num_str.replace(",", ""))
            if unit == "K":
                num /= 1000
            elif unit == "B":
                num *= 1000
            values.append(num)
        except ValueError:
            continue

    if not values:
        return 0.0
    return max(values)


def get_best_11(
    squad_df: pd.DataFrame,
    formation: list[dict],
    profile: Profile,
) -> list[dict]:
    """Assign squad players to formation slots using greedy max-weight matching.

    A formation is a list of slots, each with a position label and role_id.
    The same role_id can appear in multiple slots (e.g. two WBA slots).
    Each player is assigned to at most one slot.

    Args:
        squad_df: DataFrame of squad players with attributes
        formation: List of {"pos": "GK", "role": "sks"} dicts
        profile: weighting profile to apply

    Returns:
        List of {"pos", "role_id", "role_name", "player_idx", "player_name",
        "score", "position"} dicts, one per slot. Unfilled slots have
        player_name="(no one)" and score=0.0.
    """
    unique_role_ids = list({slot["role"] for slot in formation})
    scored = score_all_roles(squad_df, unique_role_ids, profile)
    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))

    # Build all (player_idx, slot_index, score) candidates
    # Only include players who can play the slot's role position
    candidates: list[tuple[int, int, float]] = []
    for slot_idx, slot in enumerate(formation):
        role_id = slot["role"]
        if role_id not in scored.columns:
            continue
        for player_idx in scored.index:
            player_pos = str(pos_col.loc[player_idx])
            if not player_can_play_role(player_pos, role_id):
                continue
            score = float(scored.at[player_idx, role_id])
            candidates.append((player_idx, slot_idx, score))

    # Sort by score descending — assign greedily
    candidates.sort(key=lambda x: x[2], reverse=True)

    assigned_players: set[int] = set()
    assigned_slots: set[int] = set()
    slot_assignment: dict[int, tuple[int, float]] = {}

    for player_idx, slot_idx, score in candidates:
        if player_idx in assigned_players or slot_idx in assigned_slots:
            continue
        slot_assignment[slot_idx] = (player_idx, score)
        assigned_players.add(player_idx)
        assigned_slots.add(slot_idx)
        if len(assigned_slots) == len(formation):
            break

    # Build result
    results = []
    for slot_idx, slot in enumerate(formation):
        role_id = slot["role"]
        if slot_idx in slot_assignment:
            player_idx, score = slot_assignment[slot_idx]
            results.append({
                "pos": slot["pos"],
                "role_id": role_id,
                "role_name": ROLES[role_id].name,
                "player_idx": player_idx,
                "player_name": str(name_col.loc[player_idx]),
                "score": score,
                "position": str(pos_col.loc[player_idx]),
            })
        else:
            results.append({
                "pos": slot["pos"],
                "role_id": role_id,
                "role_name": ROLES[role_id].name,
                "player_idx": -1,
                "player_name": "(no one)",
                "score": 0.0,
                "position": "",
            })

    return results


def get_formation_benchmarks(
    squad_df: pd.DataFrame,
    formation: list[dict],
    profile: Profile,
) -> list[dict]:
    """Get per-slot benchmarks from the Best 11 assignment.

    Each slot's benchmark is the player assigned to it in the Best 11.
    Used for upgrade comparison — a target is an upgrade for a slot if
    their score for that slot's role beats the assigned player's score.

    Args:
        squad_df: DataFrame of squad players
        formation: List of {"pos": "GK", "role": "sks"} dicts
        profile: weighting profile

    Returns:
        List of benchmark dicts with slot info + assigned player + score
    """
    return get_best_11(squad_df, formation, profile)


def filter_formation_upgrades(
    targets_df: pd.DataFrame,
    squad_df: pd.DataFrame,
    formation: list[dict],
    profile: Profile,
    min_margin: float = 0.0,
    max_age: int = 99,
    max_value: str = "",
    position_mode: str = "can_play",
) -> pd.DataFrame:
    """Find transfer targets who upgrade any formation slot.

    For each target, finds the formation slot where they'd provide the
    largest upgrade over the current Best 11 player in that slot.

    Args:
        targets_df: DataFrame of transfer targets
        squad_df: DataFrame of current squad
        formation: List of {"pos", "role"} slot dicts
        profile: weighting profile
        min_margin: minimum upgrade margin
        max_age: max age filter
        max_value: max transfer value filter
        position_mode: "any", "can_play", or "cannot_play"

    Returns:
        DataFrame of upgrade targets sorted by best upgrade margin
    """
    benchmarks = get_formation_benchmarks(squad_df, formation, profile)
    unique_role_ids = list({slot["role"] for slot in formation})
    scored = score_all_roles(targets_df, unique_role_ids, profile)

    if not unique_role_ids:
        return scored

    # Position filtering
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    if position_mode == "can_play":
        for role_id in unique_role_ids:
            mask = pos_col.apply(lambda p: player_can_play_role(str(p), role_id))
            scored.loc[~mask, role_id] = -999
    elif position_mode == "cannot_play":
        for role_id in unique_role_ids:
            mask = pos_col.apply(lambda p: player_can_play_role(str(p), role_id))
            scored.loc[mask, role_id] = -999

    # For each target, find the best slot to upgrade
    # A slot is defined by (pos, role_id, benchmark_score, benchmark_player)
    best_upgrade_score = pd.Series([-999.0] * len(scored), index=scored.index)
    best_upgrade_margin = pd.Series([-999.0] * len(scored), index=scored.index)
    best_upgrade_slot = pd.Series([""] * len(scored), index=scored.index)
    best_upgrade_role = pd.Series([""] * len(scored), index=scored.index)
    best_upgrade_pos = pd.Series([""] * len(scored), index=scored.index)
    squad_player_beaten = pd.Series([""] * len(scored), index=scored.index)

    for slot_idx, bm in enumerate(benchmarks):
        role_id = bm["role_id"]
        if role_id not in scored.columns:
            continue
        bm_score = bm["score"]
        bm_player = bm["player_name"]
        slot_pos = bm["pos"]

        margins = scored[role_id] - bm_score
        # This slot is the best upgrade for a target if margin > their current best
        is_better = margins > best_upgrade_margin
        for idx in scored.index:
            if is_better.loc[idx] and scored.at[idx, role_id] > 0:
                best_upgrade_score.loc[idx] = scored.at[idx, role_id]
                best_upgrade_margin.loc[idx] = margins.loc[idx]
                best_upgrade_slot.loc[idx] = f"{slot_pos} ({role_id})"
                best_upgrade_role.loc[idx] = ROLES[role_id].name
                best_upgrade_pos.loc[idx] = slot_pos
                squad_player_beaten.loc[idx] = bm_player

    scored["Target Best Score"] = best_upgrade_score.round(1)
    scored["Upgrade Margin"] = best_upgrade_margin.round(1)
    scored["Upgrade Slot"] = best_upgrade_slot
    scored["Upgrade Position"] = best_upgrade_pos
    scored["Upgrade Role"] = best_upgrade_role
    scored["Squad Player Beaten"] = squad_player_beaten

    # Filter: must be an upgrade
    mask = (scored["Upgrade Margin"] > min_margin) & (scored["Target Best Score"] > 0)
    scored = scored[mask]

    # Filter: age
    if "Age" in scored.columns:
        scored["Age"] = pd.to_numeric(scored["Age"], errors="coerce").fillna(99)
        scored = scored[scored["Age"] <= max_age]

    # Filter: transfer value
    if max_value and "Transfer Value" in scored.columns:
        max_val_num = _parse_value_string(max_value)
        if max_val_num > 0:
            scored["_value_num"] = scored["Transfer Value"].apply(_parse_value_string)
            scored = scored[scored["_value_num"] <= max_val_num]
            scored = scored.drop(columns=["_value_num"])

    scored = scored.sort_values("Upgrade Margin", ascending=False)
    return scored
