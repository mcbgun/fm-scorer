"""Sell recommendation engine: depth charts, squad status, sell priorities.

Builds on the formation-based scoring to answer:
  - Who should I sell?
  - Who should I loan out?
  - Who should I promote to the first team?
  - Who is surplus to requirements?

Key concepts:
  - Depth chart: all players ranked per formation role.
  - Squad status: Starter, Backup, Surplus, Prospect, Stagnant, Declining.
  - Sell priority: a single score ranking who to sell first, factoring in
    surplus gap, age, transfer value, role congestion, and personality.
"""

import re

import pandas as pd

from roles import ROLES
from profiles import Profile
from scorer import score_all_roles, get_best_11, _parse_value_string
from youth import compute_potential_score, get_personality_multiplier
from positions import player_can_play_role


def build_depth_chart(
    squad_df: pd.DataFrame,
    formation_slots: list[dict],
    profile: Profile,
    scored: pd.DataFrame | None = None,
) -> dict[str, list[dict]]:
    """Rank all players per formation role, creating a depth chart.

    Args:
        squad_df: DataFrame of all squad players
        formation_slots: List of {"pos", "role"} dicts
        profile: weighting profile
        scored: Pre-computed scored DataFrame (avoids re-scoring if already done)

    Returns:
        {role_id: [{"name", "age", "score", "position", "personality",
                     "transfer_value", "wage", "idx"}, ...]} sorted by score desc
    """
    unique_role_ids = list({slot["role"] for slot in formation_slots})
    if scored is None:
        scored = score_all_roles(squad_df, unique_role_ids, profile)

    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))
    age_col = pd.to_numeric(scored.get("Age", pd.Series([99] * len(scored))), errors="coerce").fillna(99)
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    pers_col = scored.get("Personality", pd.Series([""] * len(scored)))
    val_col = scored.get("Transfer Value", pd.Series([""] * len(scored)))
    wage_col = scored.get("Wage", pd.Series([""] * len(scored)))

    depth: dict[str, list[dict]] = {}
    for role_id in unique_role_ids:
        if role_id not in scored.columns:
            continue
        players = []
        for idx in scored.index:
            # Only include players who can actually play the role's position
            if not player_can_play_role(str(pos_col.loc[idx]), role_id):
                continue
            players.append({
                "name": str(name_col.loc[idx]),
                "age": int(age_col.loc[idx]),
                "score": round(float(scored.at[idx, role_id]), 1),
                "position": str(pos_col.loc[idx]),
                "personality": str(pers_col.loc[idx]),
                "transfer_value": str(val_col.loc[idx]),
                "wage": str(wage_col.loc[idx]),
                "idx": idx,
            })
        players.sort(key=lambda p: p["score"], reverse=True)
        depth[role_id] = players

    return depth


def get_starters(depth: dict[str, list[dict]], formation_slots: list[dict]) -> dict[str, dict]:
    """Identify the best 11 starters using greedy assignment.

    Returns:
        {role_id: {"name", "score", "idx"}} for each assigned slot.
        If a role appears in multiple slots, the top N players at that role
        are assigned (e.g. two WBA slots = top 2 WBA players).
    """
    # Count how many slots need each role
    role_slot_count: dict[str, int] = {}
    for slot in formation_slots:
        r = slot["role"]
        role_slot_count[r] = role_slot_count.get(r, 0) + 1

    # Greedy: assign players to slots by highest score
    assigned_players: set[int] = set()
    starters: dict[str, dict] = {}

    # Build all (player, role, score) candidates
    candidates = []
    for role_id, players in depth.items():
        for p in players:
            candidates.append((p["idx"], role_id, p["score"], p))
    candidates.sort(key=lambda c: c[2], reverse=True)

    slots_needed = dict(role_slot_count)
    for idx, role_id, score, p in candidates:
        if idx in assigned_players:
            continue
        if slots_needed.get(role_id, 0) <= 0:
            continue
        starters[f"{role_id}_{role_slot_count[role_id] - slots_needed[role_id]}"] = {
            "name": p["name"],
            "score": score,
            "idx": idx,
            "role_id": role_id,
        }
        assigned_players.add(idx)
        slots_needed[role_id] -= 1

    return starters


def classify_player(
    player: dict,
    role_id: str,
    depth: dict[str, list[dict]],
    starters: dict[str, dict],
    formation_slot_count: dict[str, int],
    max_youth_age: int = 21,
    in_squad_25: bool = True,
    squad_size_limit: int = 25,
) -> tuple[str, str, str]:
    """Classify a player and generate a recommendation.

    Uses a 25-man squad limit. Youth prospects don't count against the 25
    (FM U21 registration rules). Anyone outside the 25 who isn't a prospect
    should be sold.

    Returns:
        (status, recommendation, reason)
    """
    age = player["age"]
    score = player["score"]
    personality = player["personality"]

    # Find this player's rank at their best role
    role_players = depth.get(role_id, [])
    rank = next((i + 1 for i, p in enumerate(role_players) if p["idx"] == player["idx"]), 999)

    # Find the starter threshold for this role (worst starter's score)
    # For multi-slot roles, the threshold is the lowest-scored starter
    role_starters = [s for s in starters.values() if s["role_id"] == role_id]
    starter_score = min((s["score"] for s in role_starters), default=None)

    slots_for_role = formation_slot_count.get(role_id, 1)
    gap = (starter_score - score) if starter_score is not None else 0.0

    # Compute potential score for youth
    potential = compute_potential_score(score, age, personality)
    pers_mult = get_personality_multiplier(personality)

    # --- Classify ---

    # Starter: within the starting slots
    if rank <= slots_for_role:
        return ("Starter", "Keep — Key", f"Starting {ROLES[role_id].name} (rank {rank} of {len(role_players)})")

    # Youth classifications (age <= max_youth_age)
    if age <= max_youth_age:
        # Stagnant: low potential or bad personality — sell
        if potential < 12.0 or (pers_mult < 0.95 and age >= 19):
            return ("Stagnant", "Sell", f"Low potential ({potential:.1f}), {personality} personality, unlikely to make it")

        # Not a real prospect and outside the 25-man squad — sell or loan
        if not in_squad_25:
            if gap <= 0:
                return ("Prospect", "Promote", f"Already outperforms senior starter for {ROLES[role_id].name}, promote to squad")
            if gap <= 5.0 and pers_mult >= 1.0:
                return ("Surplus", "Loan", f"{gap:.1f} behind starter, outside {squad_size_limit}-man squad, needs game time")
            return ("Surplus", "Sell", f"Outside {squad_size_limit}-man squad, {gap:.1f} behind starter, not a real prospect")

        # In the 25-man squad as a youth player
        if gap <= 0:
            return ("Prospect", "Promote", f"Already outperforms senior starter for {ROLES[role_id].name}")
        elif gap <= 2.0:
            return ("Prospect", "Keep — Prospect", f"{gap:.1f} behind starter, potential {potential:.1f}, close to first team")
        elif gap <= 5.0:
            return ("Prospect", "Loan", f"{gap:.1f} behind starter, needs game time to develop")
        else:
            return ("Prospect", "Keep — Prospect", f"Long-term project ({gap:.1f} gap), potential {potential:.1f}")

    # Senior classifications (age > max_youth_age)
    # Anyone outside the 25-man squad who isn't a youth prospect → Sell
    if not in_squad_25:
        if age >= 28:
            return ("Surplus", "Sell", f"Age {age}, outside 25-man squad, {gap:.1f} behind starter, value declining")
        return ("Surplus", "Sell", f"Outside 25-man squad, {gap:.1f} behind starter for {ROLES[role_id].name}")

    # In the 25-man squad but not starting
    # Aging players (29+) who aren't close to starting should be sold —
    # they're blocking squad spots that could go to youth prospects
    if age >= 29 and gap > 1.5:
        return ("Declining", "Sell", f"Age {age}, in {squad_size_limit}-man squad but {gap:.1f} behind starter, blocking youth development")
    if age >= 30 and gap > 0.5:
        return ("Declining", "Sell", f"Age {age}, in {squad_size_limit}-man squad but {gap:.1f} behind starter, value declining, sell now")
    if age >= 28:
        return ("Backup", "Keep — Backup", f"Age {age}, in {squad_size_limit}-man squad, {gap:.1f} behind starter, experienced cover")

    return ("Backup", "Keep — Backup", f"In {squad_size_limit}-man squad, {gap:.1f} behind starter, squad depth")


def compute_sell_priority(
    player: dict,
    role_id: str,
    depth: dict[str, list[dict]],
    starters: dict[str, dict],
    formation_slot_count: dict[str, int],
) -> float:
    """Compute a sell priority score. Higher = sell first.

    Factors:
      - Surplus gap (bigger gap = more sellable)
      - Age decline risk (older = sell sooner before value drops)
      - Transfer value (high value surplus = capitalize now)
      - Role congestion (more players ahead = more sellable)
      - Personality penalty (youth with bad personality = sell boost)
      - Potential offset (youth with high potential = sell penalty)
    """
    age = player["age"]
    score = player["score"]
    personality = player["personality"]

    # Find starter threshold (worst starter's score for multi-slot roles)
    role_starters = [s for s in starters.values() if s["role_id"] == role_id]
    starter_score = min((s["score"] for s in role_starters), default=None)
    gap = (starter_score - score) if starter_score is not None else 0.0

    # Surplus gap factor (only positive gaps count)
    surplus_factor = max(0, gap) * 2.0

    # Age decline risk
    age_factor = max(0, age - 24) * 0.5

    # Transfer value factor (log scale — £10M is more sellable than £100K)
    val_num = _parse_value_string(player["transfer_value"])
    val_factor = 0.0
    if val_num > 0:
        import math
        val_factor = math.log10(val_num + 1) * 0.5

    # Role congestion: how many players are ahead at this role?
    role_players = depth.get(role_id, [])
    rank = next((i + 1 for i, p in enumerate(role_players) if p["idx"] == player["idx"]), 999)
    slots_for_role = formation_slot_count.get(role_id, 1)
    players_ahead = max(0, rank - slots_for_role)
    congestion_factor = players_ahead * 0.3

    # Personality penalty for youth
    pers_mult = get_personality_multiplier(personality)
    pers_penalty = 0.0
    if age <= 21 and pers_mult < 1.0:
        pers_penalty = (1.0 - pers_mult) * 3.0

    # Potential offset for youth (high potential = don't sell)
    potential_offset = 0.0
    if age <= 21:
        potential = compute_potential_score(score, age, personality)
        potential_offset = -potential * 0.3

    return round(surplus_factor + age_factor + val_factor + congestion_factor + pers_penalty + potential_offset, 2)


def generate_sell_recommendations(
    squad_df: pd.DataFrame,
    formation_slots: list[dict],
    profile: Profile,
    max_youth_age: int = 21,
    squad_size_limit: int = 25,
) -> pd.DataFrame:
    """Generate sell/loan/keep recommendations for all squad players.

    Uses a squad size limit (default 25). Youth prospects (age <= max_youth_age
    with potential >= 12 and decent personality) don't count against the limit.
    Any senior player outside the best 25 who isn't a prospect is flagged to sell.

    Args:
        squad_df: DataFrame of all squad players
        formation_slots: List of {"pos", "role"} dicts from saved formation
        profile: weighting profile
        max_youth_age: age threshold for youth classification
        squad_size_limit: max registered squad size (FM default 25)

    Returns:
        DataFrame with columns:
          - Info (Name, Age, Position, Personality, Transfer Value, Wage)
          - Best Formation Role, Role Score, Rank at Role
          - Status, Recommendation, Sell Priority, Reason
    """
    if not formation_slots:
        formation_slots = [{"pos": "GK", "role": "gkd"}]

    # Score once and reuse for depth chart + classification
    unique_role_ids = list({slot["role"] for slot in formation_slots})
    scored = score_all_roles(squad_df, unique_role_ids, profile)

    # Build depth chart (reuse scored DataFrame)
    depth = build_depth_chart(squad_df, formation_slots, profile, scored=scored)

    # Get starters (best 11)
    starters = get_starters(depth, formation_slots)

    # Count slots per role
    role_slot_count: dict[str, int] = {}
    for slot in formation_slots:
        r = slot["role"]
        role_slot_count[r] = role_slot_count.get(r, 0) + 1

    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))
    age_col = pd.to_numeric(scored.get("Age", pd.Series([99] * len(scored))), errors="coerce").fillna(99)
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    pers_col = scored.get("Personality", pd.Series([""] * len(scored)))
    val_col = scored.get("Transfer Value", pd.Series([""] * len(scored)))
    wage_col = scored.get("Wage", pd.Series([""] * len(scored)))

    # First pass: compute best role, score, potential for every player
    all_players = []
    for idx in scored.index:
        player_pos = str(pos_col.loc[idx])
        best_role_id = None
        best_score = -999
        for role_id in unique_role_ids:
            if role_id in scored.columns and player_can_play_role(player_pos, role_id):
                s = float(scored.at[idx, role_id])
                if s > best_score:
                    best_score = s
                    best_role_id = role_id
        if best_role_id is None:
            continue

        age = int(age_col.loc[idx])
        personality = str(pers_col.loc[idx])
        potential = compute_potential_score(best_score, age, personality)
        pers_mult = get_personality_multiplier(personality)

        all_players.append({
            "idx": idx,
            "name": str(name_col.loc[idx]),
            "age": age,
            "score": round(best_score, 1),
            "position": str(pos_col.loc[idx]),
            "personality": personality,
            "transfer_value": str(val_col.loc[idx]),
            "wage": str(wage_col.loc[idx]),
            "best_role_id": best_role_id,
            "potential": potential,
            "pers_mult": pers_mult,
        })

    # Determine which players are youth prospects (don't count against 25-man limit)
    # Strict criteria: a real prospect must be close to the first team (gap <= 3)
    # with good potential and personality, OR be very young (<=17) with high
    # potential (>= 14). Players who are 3+ points behind with average potential
    # are NOT exempt — they count against the squad limit and should be loaned/sold.
    youth_prospects = set()
    for p in all_players:
        if p["age"] > max_youth_age:
            continue
        # Find the starter threshold for this player's best role
        role_starters = [s for s in starters.values() if s["role_id"] == p["best_role_id"]]
        starter_score = min((s["score"] for s in role_starters), default=None)
        gap = (starter_score - p["score"]) if starter_score is not None else 0.0

        is_real_prospect = (
            p["pers_mult"] >= 1.0
            and p["potential"] >= 13.0
            and (
                gap <= 3.0
                or (p["potential"] >= 14.0 and p["age"] <= 17)
            )
        )
        if is_real_prospect:
            youth_prospects.add(p["idx"])

    # Build the 25-man squad from non-prospects, ranked by best role score
    non_prospects = [p for p in all_players if p["idx"] not in youth_prospects]
    non_prospects.sort(key=lambda p: p["score"], reverse=True)
    squad_25_ids = {p["idx"] for p in non_prospects[:squad_size_limit]}

    rows = []
    for p in all_players:
        idx = p["idx"]
        role_id = p["best_role_id"]

        # Find rank at best role
        role_players = depth.get(role_id, [])
        rank = next((i + 1 for i, rp in enumerate(role_players) if rp["idx"] == idx), 999)

        in_squad_25 = idx in squad_25_ids

        status, recommendation, reason = classify_player(
            p, role_id, depth, starters, role_slot_count, max_youth_age, in_squad_25, squad_size_limit
        )

        sell_priority = compute_sell_priority(
            p, role_id, depth, starters, role_slot_count
        )

        rows.append({
            "Name": p["name"],
            "Age": p["age"],
            "Position": p["position"],
            "Personality": p["personality"],
            "Transfer Value": p["transfer_value"],
            "Wage": p["wage"],
            "Best Formation Role": ROLES[role_id].name,
            "Role Score": p["score"],
            "Rank at Role": rank,
            "Status": status,
            "Recommendation": recommendation,
            "Sell Priority": sell_priority,
            "Reason": reason,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Sort: Sell first (by priority desc), then Loan, then Promote, then Keep
    rec_order = {"Sell": 0, "Loan": 1, "Promote": 2, "Keep — Backup": 3, "Keep — Prospect": 4, "Keep — Key": 5}
    result["_rec_order"] = result["Recommendation"].map(lambda r: rec_order.get(r, 9))
    result = result.sort_values(["_rec_order", "Sell Priority"], ascending=[True, False])
    result = result.drop(columns=["_rec_order"])

    return result.reset_index(drop=True)


# --- Squad Registration Optimizer ---

def _is_hgn(hg_status: str) -> bool:
    """Check if a player's Home-Grown Status qualifies as Home Grown Nation."""
    return hg_status in (
        "Trained at club (0-21)",
        "Trained in nation (0-21)",
        "Trained in nation (15-21)",
    )


def _parse_dob(dob_str: str) -> tuple[int, int, int] | None:
    """Parse a DoB string into (day, month, year).

    Handles formats like '20/9/2009 (20 years old)', '2009-09-20', etc.
    """
    s = str(dob_str).strip()
    # Try DD/MM/YYYY or D/M/YYYY (FM default)
    m = re.match(r"(\d+)/(\d+)/(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Try ISO format YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1))
    return None


def _is_u21_exempt(dob_str: str, ref_year: int = 2029) -> bool:
    """Check if a player is U21-exempt (age <= 20 on 1 Jan of ref_year).

    Uses the DoB column from the FM export to determine exact eligibility.
    FM's rule: players aged 20 or younger on 1 January don't need registration.
    """
    from datetime import date
    parsed = _parse_dob(dob_str)
    if not parsed:
        return False
    d, m, y = parsed
    try:
        birth = date(y, m, d)
        ref = date(ref_year, 1, 1)
        age_days = (ref - birth).days
        age_years = age_days / 365.25
        return age_years <= 20.0
    except ValueError:
        return False


def optimize_squad_registration(
    squad_df: pd.DataFrame,
    formation_slots: list[dict],
    profile: Profile,
    max_squad: int = 25,
    min_squad: int = 15,
    min_hgn: int = 8,
    u21_age: int = 20,
) -> dict:
    """Optimize the 25-man squad registration per Premier Division rules.

    Rules:
      - Players aged <= 20 on 1 Jan of the current season don't need
        registration (U21 exempt). Determined from the DoB column.
      - Max max_squad registered players (default 25).
      - Min min_squad registered players (default 15).
      - Min min_hgn Home Grown Nation players in the registered squad.

    The optimizer picks the best possible registered squad that maximizes
    total quality (sum of best formation role scores) while satisfying
    all constraints. Players who don't make the 25 are flagged for
    sell/loan.

    Returns:
        dict with keys:
          - "registered": list of player dicts in the optimal 25
          - "unregistered": list of player dicts outside the 25
          - "u21_exempt": list of U21 players (don't need registration)
          - "hgn_count": HGN count in registered squad
          - "non_hgn_count": non-HGN count in registered squad
          - "total_quality": sum of scores in registered squad
          - "constraints_met": bool
          - "issues": list of strings describing any constraint violations
          - "dropped_for_registration": players who'd be in the best 25
            but can't be registered due to HGN constraints
    """
    unique_role_ids = list({slot["role"] for slot in formation_slots}) if formation_slots else list(ROLES.keys())
    scored = score_all_roles(squad_df, unique_role_ids, profile)

    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))
    age_col = pd.to_numeric(scored.get("Age", pd.Series([99] * len(scored))), errors="coerce").fillna(99)
    hg_col = scored.get("Home-Grown Status", pd.Series(["-"] * len(scored)))
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    val_col = scored.get("Transfer Value", pd.Series([""] * len(scored)))
    wage_col = scored.get("Wage", pd.Series([""] * len(scored)))
    dob_col = scored.get("DoB", pd.Series([""] * len(scored)))
    inf_col = scored.get("Inf", pd.Series([""] * len(scored)))

    # Build player list with best role score
    players = []
    for idx in scored.index:
        best_role_id = None
        best_score = -999
        for role_id in unique_role_ids:
            if role_id in scored.columns:
                s = float(scored.at[idx, role_id])
                if s > best_score:
                    best_score = s
                    best_role_id = role_id
        if best_role_id is None:
            continue

        age = int(age_col.loc[idx])
        hg = str(hg_col.loc[idx])
        inf_val = str(inf_col.loc[idx]) if idx in inf_col.index else ""

        # Determine HGN and U21 status:
        # - If the Inf column has U21/HGN icons (from a registration view merge),
        #   use those — they're the game's own classification.
        # - Otherwise fall back to Home-Grown Status column for HGN and DoB for U21.
        if inf_val == "U21":
            is_u21 = True
            is_hgn = _is_hgn(hg)  # Still check HG status for HGN badge
        elif inf_val == "HGN":
            is_u21 = _is_u21_exempt(str(dob_col.loc[idx])) if idx in dob_col.index else False
            is_hgn = True
        else:
            # Fall back to DoB-based U21 and HG Status for HGN
            is_hgn = _is_hgn(hg)
            dob_str = str(dob_col.loc[idx]) if idx in dob_col.index else ""
            is_u21 = _is_u21_exempt(dob_str)

        players.append({
            "idx": idx,
            "name": str(name_col.loc[idx]),
            "age": age,
            "score": round(best_score, 1),
            "position": str(pos_col.loc[idx]),
            "hg_status": hg,
            "is_hgn": is_hgn,
            "is_u21": is_u21,
            "inf": inf_val,
            "transfer_value": str(val_col.loc[idx]),
            "wage": str(wage_col.loc[idx]),
            "best_role": ROLES[best_role_id].name if best_role_id in ROLES else best_role_id,
        })

    # Split into U21-exempt and registration-required
    u21_exempt = [p for p in players if p["is_u21"]]
    reg_required = [p for p in players if not p["is_u21"]]

    # Sort registration-required by score descending
    reg_required.sort(key=lambda p: p["score"], reverse=True)

    # Optimization: pick the best 25 from reg_required, ensuring >= min_hgn HGN
    # Strategy:
    #   1. Sort all reg_required by score
    #   2. Take top 25 — check if HGN >= 8
    #   3. If not, swap lowest-scored non-HGN for next best HGN not in squad
    #   4. Repeat until constraints met or impossible

    hgn_pool = [p for p in reg_required if p["is_hgn"]]
    non_hgn_pool = [p for p in reg_required if not p["is_hgn"]]

    # Greedy: start with top 25 by score
    top_25 = reg_required[:max_squad]
    registered = list(top_25)

    # Check HGN constraint
    current_hgn = sum(1 for p in registered if p["is_hgn"])

    while current_hgn < min_hgn:
        # Find the lowest-scored non-HGN in the registered squad
        non_hgn_in_squad = [p for p in registered if not p["is_hgn"]]
        if not non_hgn_in_squad:
            break  # Can't swap anymore

        non_hgn_in_squad.sort(key=lambda p: p["score"])
        worst_non_hgn = non_hgn_in_squad[0]

        # Find the best HGN not in the registered squad
        registered_ids = {p["idx"] for p in registered}
        available_hgn = [p for p in hgn_pool if p["idx"] not in registered_ids]
        if not available_hgn:
            break  # Not enough HGN players

        available_hgn.sort(key=lambda p: p["score"], reverse=True)
        best_available_hgn = available_hgn[0]

        # Only swap if the HGN replacement isn't much worse than the non-HGN
        # (avoid severely downgrading the squad just to meet HGN quota)
        if best_available_hgn["score"] >= worst_non_hgn["score"] - 2.0:
            registered.remove(worst_non_hgn)
            registered.append(best_available_hgn)
        current_hgn = sum(1 for p in registered if p["is_hgn"])

    # Sort registered by score for display
    registered.sort(key=lambda p: p["score"], reverse=True)

    # Unregistered = reg_required not in the 25
    registered_ids = {p["idx"] for p in registered}
    unregistered = [p for p in reg_required if p["idx"] not in registered_ids]
    unregistered.sort(key=lambda p: p["score"], reverse=True)

    # Check constraints
    hgn_count = sum(1 for p in registered if p["is_hgn"])
    non_hgn_count = len(registered) - hgn_count
    total_quality = sum(p["score"] for p in registered)

    issues = []
    if len(registered) > max_squad:
        issues.append(f"Too many registered players: {len(registered)} > {max_squad}")
    if len(registered) < min_squad:
        issues.append(f"Too few registered players: {len(registered)} < {min_squad}")
    if hgn_count < min_hgn:
        issues.append(f"Not enough HGN players: {hgn_count} < {min_hgn}. Need {min_hgn - hgn_count} more home-grown signings.")
    constraints_met = len(issues) == 0

    return {
        "registered": registered,
        "unregistered": unregistered,
        "u21_exempt": u21_exempt,
        "hgn_count": hgn_count,
        "non_hgn_count": non_hgn_count,
        "total_quality": round(total_quality, 1),
        "constraints_met": constraints_met,
        "issues": issues,
        "max_squad": max_squad,
        "min_squad": min_squad,
        "min_hgn": min_hgn,
        "u21_age": u21_age,
    }
