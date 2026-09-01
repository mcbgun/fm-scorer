"""Transfer strategy engine: optimal squad planning with budget constraints.

Given the current squad, scouted targets, and budget, generates a single
optimal transfer plan that maximizes multi-season squad quality.

Algorithm:
  1. Score all squad players and targets against formation roles.
  2. Project quality over N seasons (age-based growth/decline).
  3. Parse transfer values (low end for sells, high end for buys).
  4. Parse wages.
  5. Identify sellable players (surplus, aging, replaceable starters).
  6. Identify upgrade targets (higher projected quality at a role).
  7. Identify investment targets (buy-to-sell, buy-to-loan).
  8. Run greedy optimization: pick the best value-for-money actions
     that fit within budget, considering sells to fund buys.
  9. Generate the plan with reasoning.
"""

import re
import pandas as pd

from roles import ROLES
from profiles import Profile
from scorer import score_all_roles
from youth import compute_potential_score, get_personality_multiplier
from projection import project_score
from budget import optimize_slider
from positions import player_can_play_role


def parse_value_low(val: str) -> float:
    """Parse transfer value string, return LOWER bound in millions."""
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
    return min(values) if values else 0.0


def parse_value_high(val: str) -> float:
    """Parse transfer value string, return UPPER bound in millions."""
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
    return max(values) if values else 0.0


def parse_wage(wage_str: str) -> float:
    """Parse wage string like '£19,000 p/w' into weekly wage in thousands."""
    if not wage_str or pd.isna(wage_str):
        return 0.0
    s = str(wage_str).strip()
    if s == "-" or s == "":
        return 0.0
    matches = re.findall(r"[\$£€]?([\d.,]+)\s*([KMB]?)", s)
    for num_str, unit in matches:
        try:
            num = float(num_str.replace(",", ""))
            if unit == "K":
                return num
            elif unit == "M":
                return num * 1000
            return num / 1000  # raw pounds -> thousands
        except ValueError:
            continue
    return 0.0


def _best_role_and_score(scored: pd.DataFrame, role_ids: list[str], idx: int) -> tuple[str, float]:
    """Find a player's best formation role and score, respecting position compatibility."""
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    player_pos = str(pos_col.loc[idx])
    best_role = None
    best_score = -999
    for role_id in role_ids:
        if role_id in scored.columns and player_can_play_role(player_pos, role_id):
            s = float(scored.at[idx, role_id])
            if s > best_score:
                best_score = s
                best_role = role_id
    return best_role, round(best_score, 1)


def generate_strategy(
    squad_df: pd.DataFrame,
    targets_df: pd.DataFrame | None,
    formation_slots: list[dict],
    profile: Profile,
    transfer_budget: float = 0.0,
    wage_budget: float = 0.0,
    seasons: int = 3,
    max_transfers: int = 10,
    locked_players: set[str] | None = None,
    u21_age: int = 20,
    total_wage_budget: float = 0.0,
    board_slider_cap: float = 1.0,
    current_wage_spend: float = 0.0,
    max_transfer_budget: float = 0.0,
    board_sales_percentage: float = 100.0,
) -> dict:
    """Generate the optimal transfer plan.

    Args:
        squad_df: Current squad DataFrame
        targets_df: Scouted transfer targets (None if no targets uploaded)
        formation_slots: Formation slots [{"pos", "role"}]
        profile: Weighting profile
        transfer_budget: Current transfer budget in £M (at current slider position)
        wage_budget: Additional wage budget in £K/week for buys (usually 0)
        seasons: Number of seasons to project
        max_transfers: Maximum number of transfers per phase (sells and buys
            each get up to this many actions)
        locked_players: Set of player names to never sell
        u21_age: U21 exemption age cutoff
        total_wage_budget: Current wage budget in £K/week (at current slider position)
        board_slider_cap: Max fraction of slider range board allows (0-1)
        current_wage_spend: Actual in-game weekly wage spend in £K/week
        max_transfer_budget: Transfer budget at slider far right / position 10 in £M
        board_sales_percentage: Percentage of sale proceeds the board makes
            available for transfers (default 100, often 70-80 in-game)

    Returns:
        dict with keys:
          - diagnosis: current squad state
          - actions: list of recommended actions (sell/buy/promote/loan)
          - financials: budget summary
          - projected_quality: current vs projected squad quality
          - budget_slider: slider optimization result (or None)
          - reasoning: explanation of the plan
    """
    if not formation_slots:
        formation_slots = [{"pos": "GK", "role": "gkd"}]

    locked = locked_players or set()
    unique_role_ids = list({slot["role"] for slot in formation_slots})
    role_slot_count: dict[str, int] = {}
    for slot in formation_slots:
        r = slot["role"]
        role_slot_count[r] = role_slot_count.get(r, 0) + 1

    # Score squad and targets
    squad_scored = score_all_roles(squad_df, unique_role_ids, profile)
    target_scored = score_all_roles(targets_df, unique_role_ids, profile) if targets_df is not None and not targets_df.empty else pd.DataFrame()

    # Build squad player list with projections
    squad_players = _build_player_list(squad_scored, unique_role_ids, seasons, is_squad=True)
    target_players = _build_player_list(target_scored, unique_role_ids, seasons, is_squad=False) if not target_scored.empty else []

    # Current squad diagnosis
    starters = _find_starters(squad_players, role_slot_count)
    current_quality = sum(p["avg_projected"] for p in starters)
    current_wages = sum(p["wage"] for p in squad_players)
    surplus = [p for p in squad_players if p["name"] not in {s["name"] for s in starters} and p["name"] not in locked]
    gaps = _identify_gaps(starters, role_slot_count, threshold=12.0)

    # Identify sellable players
    sellable = []
    for p in squad_players:
        if p["name"] in locked:
            continue
        reason = _sell_reason(p, starters, squad_players)
        if reason:
            sellable.append({**p, "sell_reason": reason})

    # Identify buy targets (exclude players already in the squad)
    squad_names = {p["name"] for p in squad_players}
    buy_candidates = []
    for t in target_players:
        if t["name"] in squad_names:
            continue
        upgrade = _evaluate_target(t, starters, role_slot_count, seasons)
        if upgrade:
            buy_candidates.append({**t, **upgrade})

    # Sort buy candidates: prioritize real upgrades over investments,
    # then by value-for-money (quality gain per £M)
    for t in buy_candidates:
        cost = t["buy_value"]
        if cost > 0:
            t["value_for_money"] = round(t["quality_gain"] / cost, 3)
        else:
            t["value_for_money"] = 99.0  # free transfer = high value but not infinite
    # Sort by: upgrade type first (upgrades > gap_fills > future > investments),
    # then by quality_gain, then by value_for_money
    type_priority = {"upgrade": 0, "gap_fill": 1, "future_upgrade": 2, "buy_to_sell": 3, "buy_to_loan": 4}
    buy_candidates.sort(key=lambda t: (
        type_priority.get(t.get("investment_type", "upgrade"), 5),
        -t["quality_gain"],
        -t["value_for_money"],
    ))

    # Sort sellable by sell priority (aging + low quality first)
    sellable.sort(key=lambda p: (-p["age"], p["avg_projected"]))

    # --- Greedy optimization ---
    # Start with current squad. Try to sell + buy to maximize quality.
    actions = []
    remaining_budget = transfer_budget
    remaining_wage_budget = wage_budget
    sold_names = set()
    bought_names = set()
    sells_made = 0
    buys_made = 0

    # Phase 1: Sell aging/surplus players to raise funds
    # Board only makes a percentage of sale proceeds available for transfers
    sales_factor = board_sales_percentage / 100.0
    sale_proceeds = 0.0          # total sale value (for display)
    available_proceeds = 0.0     # amount actually added to transfer budget
    wage_saved_total = 0.0
    for p in sellable:
        if sells_made >= max_transfers:
            break
        sell_value = p["sell_value"]
        wage_saved = p["wage"]
        # Always sell if: aging (29+), low quality, or surplus
        should_sell = (
            p["age"] >= 29
            or p["avg_projected"] < 11.0
            or "surplus" in p.get("sell_reason", "")
        )
        if should_sell:
            actions.append({
                "type": "sell",
                "name": p["name"],
                "age": p["age"],
                "role": p["best_role_name"],
                "value": sell_value,
                "wage_saved": wage_saved,
                "reason": p["sell_reason"],
            })
            proceeds_to_budget = sell_value * sales_factor
            remaining_budget += proceeds_to_budget
            remaining_wage_budget += wage_saved
            sale_proceeds += sell_value
            available_proceeds += proceeds_to_budget
            wage_saved_total += wage_saved
            sold_names.add(p["name"])
            sells_made += 1

    # Phase 1b: Optimize budget slider (wage → transfer conversion)
    squad_wage_bill = sum(p["wage"] for p in squad_players)
    slider_result = optimize_slider(
        transfer_budget=transfer_budget,
        total_wage_budget=total_wage_budget,
        squad_wage_bill=squad_wage_bill,
        sale_proceeds=available_proceeds,
        wage_saved_from_sells=wage_saved_total,
        buy_candidates=buy_candidates,
        board_slider_cap=board_slider_cap,
        current_wage_spend=current_wage_spend,
        max_transfer_budget=max_transfer_budget,
    )
    # Apply slider adjustment to budgets
    if slider_result["transfer_from_slider"] > 0:
        remaining_budget += slider_result["transfer_from_slider"]
        # Wage room for buys = headroom after slider + any additional wage_budget
        remaining_wage_budget = slider_result["wage_headroom_after"] + wage_budget

    # Phase 2: Buy upgrades (best value-for-money first)
    # Track how many buys per role to avoid overloading one position
    buys_per_role: dict[str, int] = {}
    for t in buy_candidates:
        if buys_made >= max_transfers:
            break
        role = t["best_role"]
        # Don't buy more than the formation allows for that role
        # (plus 1 backup per slot)
        max_for_role = role_slot_count.get(role, 1) + 1
        if buys_per_role.get(role, 0) >= max_for_role:
            continue
        cost = t["buy_value"]
        wage_cost = t["wage"]
        if cost > remaining_budget or wage_cost > remaining_wage_budget:
            continue
        actions.append({
            "type": "buy",
            "name": t["name"],
            "age": t["age"],
            "role": t["best_role_name"],
            "value": cost,
            "wage": wage_cost,
            "quality_gain": t["quality_gain"],
            "value_for_money": t["value_for_money"],
            "reason": t["reason"],
            "investment_type": t.get("investment_type", "upgrade"),
        })
        remaining_budget -= cost
        remaining_wage_budget -= wage_cost
        bought_names.add(t["name"])
        buys_per_role[role] = buys_per_role.get(role, 0) + 1
        buys_made += 1

    # Phase 3: Promote youth if gaps remain
    youth_promoted = []
    for p in squad_players:
        if p["name"] in sold_names or p["name"] in locked:
            continue
        if p["age"] > u21_age:
            continue
        if p["name"] in {s["name"] for s in starters}:
            continue
        # Check if this youth can fill a gap
        for gap_role in gaps:
            if p["best_role"] == gap_role and p["avg_projected"] >= 10.0:
                actions.append({
                    "type": "promote",
                    "name": p["name"],
                    "age": p["age"],
                    "role": p["best_role_name"],
                    "reason": f"Youth prospect can cover {ROLES[gap_role].name} gap",
                })
                youth_promoted.append(p["name"])
                break

    # Phase 4: Loan out remaining surplus youth
    for p in squad_players:
        if p["name"] in sold_names or p["name"] in youth_promoted or p["name"] in locked:
            continue
        if p["age"] > u21_age:
            continue
        if p["name"] in {s["name"] for s in starters}:
            continue
        # Loan if not promoted and has potential
        if p["potential"] >= 12.0 and p["age"] <= u21_age:
            actions.append({
                "type": "loan",
                "name": p["name"],
                "age": p["age"],
                "role": p["best_role_name"],
                "reason": "Loan for development — needs game time",
            })

    # Calculate projected quality after plan
    remaining_squad = [p for p in squad_players if p["name"] not in sold_names]
    # Add bought players to squad
    for t in target_players:
        if t["name"] in bought_names:
            remaining_squad.append(t)
    new_starters = _find_starters(remaining_squad, role_slot_count)
    new_quality = sum(p["avg_projected"] for p in new_starters)
    new_wages = sum(p["wage"] for p in remaining_squad)

    total_sales = sum(a["value"] for a in actions if a["type"] == "sell")
    total_buys = sum(a["value"] for a in actions if a["type"] == "buy")
    net_spend = total_buys - total_sales
    wage_change = sum(a.get("wage", 0) for a in actions if a["type"] == "buy") - sum(a.get("wage_saved", 0) for a in actions if a["type"] == "sell")

    return {
        "diagnosis": {
            "squad_size": len(squad_players),
            "current_quality": round(current_quality, 1),
            "current_wages": round(current_wages, 1),
            "num_starters": len(starters),
            "surplus_count": len(surplus),
            "gaps": gaps,
            "gaps_detail": [{"role": g, "name": ROLES[g].name} for g in gaps if g in ROLES],
        },
        "actions": actions,
        "financials": {
            "transfer_budget": transfer_budget,
            "wage_budget": wage_budget,
            "total_sales": round(total_sales, 2),
            "available_proceeds": round(available_proceeds, 2),
            "board_sales_percentage": board_sales_percentage,
            "total_buys": round(total_buys, 2),
            "net_spend": round(net_spend, 2),
            "remaining_budget": round(remaining_budget, 2),
            "wage_change": round(wage_change, 2),
            "new_wages": round(new_wages, 1),
        },
        "projected_quality": {
            "current": round(current_quality, 1),
            "projected": round(new_quality, 1),
            "improvement": round(new_quality - current_quality, 1),
            "seasons": seasons,
        },
        "budget_slider": slider_result if total_wage_budget > 0 and max_transfer_budget > 0 else None,
        "reasoning": _generate_reasoning(actions, current_quality, new_quality, gaps, sellable, buy_candidates),
    }


def _build_player_list(scored: pd.DataFrame, role_ids: list[str], seasons: int, is_squad: bool) -> list[dict]:
    """Build a list of player dicts with scores, projections, and financial info."""
    if scored.empty:
        return []

    name_col = scored.get("Name", pd.Series(["?"] * len(scored)))
    age_col = pd.to_numeric(scored.get("Age", pd.Series([99] * len(scored))), errors="coerce").fillna(99)
    pos_col = scored.get("Position", pd.Series([""] * len(scored)))
    pers_col = scored.get("Personality", pd.Series([""] * len(scored)))
    val_col = scored.get("Transfer Value", pd.Series([""] * len(scored)))
    wage_col = scored.get("Wage", pd.Series([""] * len(scored)))

    players = []
    for idx in scored.index:
        best_role, best_score = _best_role_and_score(scored, role_ids, idx)
        if best_role is None:
            continue

        age = int(age_col.loc[idx])
        personality = str(pers_col.loc[idx])
        position = str(pos_col.loc[idx])
        potential = compute_potential_score(best_score, age, personality)

        projections = project_score(best_score, age, personality, position, seasons, potential)

        # Compute per-role projected scores for proper starter assignment
        role_scores: dict[str, float] = {}
        for role_id in role_ids:
            if role_id in scored.columns and player_can_play_role(position, role_id):
                s = float(scored.at[idx, role_id])
                if s > 0:
                    role_proj = project_score(s, age, personality, position, seasons, potential)
                    role_scores[role_id] = round(sum(role_proj) / len(role_proj), 1)

        # Financial values: low end for sells, high end for buys
        val_str = str(val_col.loc[idx])
        sell_value = parse_value_low(val_str)
        buy_value = parse_value_high(val_str)
        wage = parse_wage(str(wage_col.loc[idx]))

        players.append({
            "name": str(name_col.loc[idx]),
            "age": age,
            "score": best_score,
            "position": position,
            "personality": personality,
            "best_role": best_role,
            "best_role_name": ROLES[best_role].name if best_role in ROLES else best_role,
            "potential": potential,
            "sell_value": sell_value,
            "buy_value": buy_value,
            "wage": wage,
            "current_score": best_score,
            "avg_projected": round(sum(projections) / len(projections), 1),
            "projections": projections,
            "role_scores": role_scores,
        })
    return players


def _find_starters(players: list[dict], role_slot_count: dict[str, int]) -> list[dict]:
    """Find the best starter for each formation slot using greedy max-weight matching.

    Instead of assigning each player to their single best role, this considers
    all (player, role, projected_score) combinations and assigns greedily to
    maximize total squad quality. This prevents a weak player from being
    classified as a "starter" just because no one else has that role as their
    best — a stronger player who is better at multiple roles gets assigned
    optimally.
    """
    # Build all (player, role, projected_score) candidates
    candidates: list[tuple[float, dict, str]] = []
    for p in players:
        for role_id, proj_score in p.get("role_scores", {}).items():
            candidates.append((proj_score, p, role_id))

    # Sort by projected score descending — assign greedily
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_names: set[str] = set()
    role_filled: dict[str, int] = {r: 0 for r in role_slot_count}
    starters = []

    for proj_score, player, role_id in candidates:
        if player["name"] in used_names:
            continue
        if role_filled.get(role_id, 0) >= role_slot_count.get(role_id, 0):
            continue
        # Assign this player to this role slot
        starter = {**player}
        starter["best_role"] = role_id
        starter["best_role_name"] = ROLES[role_id].name if role_id in ROLES else role_id
        starter["avg_projected"] = proj_score
        starters.append(starter)
        used_names.add(player["name"])
        role_filled[role_id] = role_filled.get(role_id, 0) + 1

    return starters


def _identify_gaps(starters: list[dict], role_slot_count: dict[str, int], threshold: float = 12.0) -> list[str]:
    """Identify roles where no starter exceeds the quality threshold."""
    gaps = []
    starter_roles = [s["best_role"] for s in starters]
    for role, count in role_slot_count.items():
        role_starters = [s for s in starters if s["best_role"] == role]
        if len(role_starters) < count:
            gaps.append(role)
        elif all(s["avg_projected"] < threshold for s in role_starters):
            gaps.append(role)
    return gaps


def _sell_reason(player: dict, starters: list[dict], all_players: list[dict]) -> str | None:
    """Determine if a player should be sold and why."""
    name = player["name"]
    is_starter = any(s["name"] == name for s in starters)

    # Aging surplus
    if player["age"] >= 30 and not is_starter:
        return f"Age {player['age']}, declining, sell before value drops (£{player['sell_value']:.1f}M)"

    # Aging starter with replacement available
    if player["age"] >= 30 and is_starter:
        # Check if there's a younger player at the same role
        same_role = [p for p in all_players if p["best_role"] == player["best_role"] and p["name"] != name and p["age"] < player["age"]]
        if same_role and max(p["avg_projected"] for p in same_role) >= player["avg_projected"] - 1.0:
            return f"Age {player['age']}, replaceable by younger option at {player['best_role_name']}"

    # Low quality surplus
    if player["avg_projected"] < 10.0 and not is_starter:
        return f"Low quality (avg projected {player['avg_projected']:.1f}), surplus at {player['best_role_name']}"

    # Surplus (many players at same role)
    if not is_starter and player["age"] >= 24:
        same_role = [p for p in all_players if p["best_role"] == player["best_role"]]
        if len(same_role) > 3:
            return f"Surplus at {player['best_role_name']} ({len(same_role)} players), sell for squad balance"

    return None


def _evaluate_target(target: dict, starters: list[dict], role_slot_count: dict[str, int], seasons: int) -> dict | None:
    """Evaluate a transfer target. Returns upgrade info or None."""
    role = target["best_role"]
    role_starters = [s for s in starters if s["best_role"] == role]

    if not role_starters:
        # No starter at this role — target fills a gap
        quality_gain = target["avg_projected"]
        return {
            "quality_gain": round(quality_gain, 1),
            "replaces": "(new role coverage)",
            "reason": f"Fills gap at {target['best_role_name']} (no current starter)",
            "investment_type": "gap_fill",
        }

    current_best = max(s["avg_projected"] for s in role_starters)
    quality_gain = target["avg_projected"] - current_best

    if quality_gain <= 0:
        # Not an upgrade — check if investment target (young + high potential)
        if target["age"] <= 19 and target["potential"] >= 14.0:
            return {
                "quality_gain": 0.5,  # Small long-term gain
                "replaces": role_starters[0]["name"],
                "reason": f"Investment: age {target['age']}, potential {target['potential']:.1f}, buy-to-loan or develop",
                "investment_type": "buy_to_loan",
            }
        return None

    # It's an upgrade
    replaces_name = min(role_starters, key=lambda s: s["avg_projected"])["name"]
    investment_type = "upgrade"
    if target["age"] <= 19 and target["potential"] >= 14.0:
        investment_type = "buy_to_sell"
        reason = f"Buy-to-sell: age {target['age']}, potential {target['potential']:.1f}, value likely to rise"
    elif target["age"] <= 21 and quality_gain < 1.0:
        investment_type = "future_upgrade"
        reason = f"Young upgrade ({quality_gain:+.1f} over {replaces_name}), age {target['age']}, will improve"
    else:
        reason = f"Upgrade ({quality_gain:+.1f} over {replaces_name}) at {target['best_role_name']}, age {target['age']}"

    return {
        "quality_gain": round(quality_gain, 1),
        "replaces": replaces_name,
        "reason": reason,
        "investment_type": investment_type,
    }


def _generate_reasoning(actions: list[dict], current_quality: float, new_quality: float, gaps: list[str], sellable: list[dict], buy_candidates: list[dict]) -> str:
    """Generate a human-readable summary of the strategy."""
    sells = [a for a in actions if a["type"] == "sell"]
    buys = [a for a in actions if a["type"] == "buy"]
    promotes = [a for a in actions if a["type"] == "promote"]
    loans = [a for a in actions if a["type"] == "loan"]

    parts = []
    if sells:
        parts.append(f"Sell {len(sells)} player(s) to raise funds and free up wages")
    if buys:
        upgrade_buys = [b for b in buys if b.get("investment_type") == "upgrade"]
        investment_buys = [b for b in buys if b.get("investment_type") in ("buy_to_sell", "buy_to_loan", "future_upgrade")]
        gap_fills = [b for b in buys if b.get("investment_type") == "gap_fill"]
        if upgrade_buys:
            parts.append(f"sign {len(upgrade_buys)} first-team upgrade(s)")
        if gap_fills:
            parts.append(f"fill {len(gap_fills)} positional gap(s)")
        if investment_buys:
            parts.append(f"invest in {len(investment_buys)} young player(s) for the future")
    if promotes:
        parts.append(f"promote {len(promotes)} youth player(s)")
    if loans:
        parts.append(f"loan out {len(loans)} developing player(s)")

    if not parts:
        return "No changes recommended — the current squad is well-balanced."

    summary = "Optimal plan: " + ", ".join(parts) + "."
    improvement = new_quality - current_quality
    if improvement > 0:
        summary += f" Projected squad quality improves by {improvement:+.1f} points over the planning horizon."
    elif improvement < 0:
        summary += f" Projected squad quality changes by {improvement:+.1f} (investment for future seasons)."
    else:
        summary += " Projected squad quality remains stable."

    return summary
