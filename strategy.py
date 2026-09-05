"""Transfer strategy engine.

Given the squad, scouted targets, formation and budgets, propose a window plan
that maximises *multi-season Best-XI quality per pound* while keeping depth.

This is an explicit greedy heuristic (not a proven optimum):

  1. Build a ``SquadAnalysis`` (side-aware optimal Best XI).
  2. Project every player's slot score over ``seasons`` (with development
     signals and a low/mid/high band) -> ``horizon`` score = mean over horizon.
  3. For every target and slot compute the projected margin over the incumbent.
  4. Repeatedly pick the candidate with the best *marginal lineup gain per £M*
     (gain re-evaluated with a fresh Hungarian assignment each step), subject to
     the transfer + wage budget, ``min_gain`` and ``min_gain_per_m`` thresholds.
     When the best candidate is unaffordable, sales are added from the sell
     list - never a starter, never someone whose exit leaves a slot without
     cover - until it is.
  5. Promote / loan youth per the sell engine.

Every number carries the ``assumption`` ("low" / "mid" / "high") used for
attribute ranges *and* transfer value ranges, so a conservative plan buys at
the top of the asking range and scores unknown attributes at their floor.
"""

import pandas as pd

from assignment import INCOMPATIBLE, assign_slots
from budget import optimize_slider
from development import extract_signals
from money import fmt_millions, fmt_wage, parse_value_range, parse_wage
from profiles import Profile
from projection import project_band
from roles import ROLES
from scorer import familiarity_matrix, score_all_roles
from sell import REC_ORDER, generate_sell_recommendations
from squad_model import SquadAnalysis
from youth import compute_potential_band

MAX_CANDIDATES_PER_SLOT = 12


def _horizon(scores: list[float]) -> float:
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _value_for_assumption(lo: float, hi: float, assumption: str, buying: bool) -> float:
    if assumption == "low":
        return hi if buying else lo
    if assumption == "high":
        return lo if buying else hi
    return (lo + hi) / 2


class _Pool:
    """Players available for lineup evaluation: squad + bought targets."""

    def __init__(self, analysis: SquadAnalysis, seasons: int, assumption: str):
        self.analysis = analysis
        self.seasons = seasons
        self.assumption = assumption
        self.formation = analysis.formation
        # horizon[(player_key, slot_idx)] -> projected slot score over horizon
        self.horizon: dict[tuple[str, int], float] = {}
        self.players: dict[str, dict] = {}
        for p in analysis.players:
            key = f"squad:{p.idx}"
            self.players[key] = {
                "key": key, "name": p.name, "age": p.age, "position": p.position, "personality": p.personality,
                "wage_k": p.wage_k, "value_lo": p.value_lo, "value_hi": p.value_hi, "is_squad": True, "idx": p.idx,
                "needs_scouting": p.needs_scouting, "record": p,
            }
            row = analysis.scored.loc[p.idx]
            sig = extract_signals(row)
            for slot_idx in range(len(self.formation)):
                cur = analysis.slot_score(slot_idx, p.idx)
                if cur <= 0:
                    continue
                lo, mid, hi = project_band(cur, p.age, p.personality, p.position, seasons, None, sig)
                self.horizon[(key, slot_idx)] = _horizon({"low": lo, "high": hi}.get(assumption, mid))

    def add_target(self, t: dict) -> None:
        self.players[t["key"]] = t
        for slot_idx, h in t["horizon_by_slot"].items():
            self.horizon[(t["key"], slot_idx)] = h

    def lineup(self, keys: list[str]) -> tuple[float, dict[int, tuple[str | None, float]]]:
        slots = list(range(len(self.formation)))

        def score_fn(slot_idx, key):
            return self.horizon.get((key, slot_idx), INCOMPATIBLE)

        result = assign_slots(slots, keys, score_fn)
        total = sum(v[1] for v in result.values() if v[0] is not None)
        return round(total, 2), result

    def cover_ok(self, keys: list[str], starters: dict[int, tuple[str | None, float]]) -> bool:
        """Every slot must still have at least one non-starting player who can play it."""
        starter_keys = {v[0] for v in starters.values() if v[0] is not None}
        for slot_idx in range(len(self.formation)):
            if not any(k not in starter_keys and (k, slot_idx) in self.horizon for k in keys):
                return False
        return True


def _build_targets(
    targets_df: pd.DataFrame,
    analysis: SquadAnalysis,
    seasons: int,
    assumption: str,
    exclude_unscouted: bool,
    squad_names: set[str],
) -> list[dict]:
    if targets_df is None or targets_df.empty:
        return []
    scored = score_all_roles(targets_df, analysis.role_ids, analysis.profile, assumption, with_bounds=True)
    if exclude_unscouted and "Needs Scouting" in scored.columns:
        scored = scored[~scored["Needs Scouting"].astype(bool)]
    if scored.empty:
        return []
    fam = familiarity_matrix(scored, analysis.formation)
    n = len(scored)
    name_col = scored["Name"].astype(str) if "Name" in scored.columns else pd.Series(["?"] * n, index=scored.index)
    age_col = pd.to_numeric(scored.get("Age", pd.Series(99, index=scored.index)), errors="coerce").fillna(99)
    pos_col = scored["Position"].astype(str) if "Position" in scored.columns else pd.Series([""] * n, index=scored.index)
    pers_col = scored["Personality"].astype(str) if "Personality" in scored.columns else pd.Series([""] * n, index=scored.index)
    val_col = scored["Transfer Value"].astype(str) if "Transfer Value" in scored.columns else pd.Series([""] * n, index=scored.index)
    wage_col = scored["Wage"].astype(str) if "Wage" in scored.columns else pd.Series([""] * n, index=scored.index)
    club_col = scored["Club"].astype(str) if "Club" in scored.columns else pd.Series([""] * n, index=scored.index)
    ns_col = scored["Needs Scouting"] if "Needs Scouting" in scored.columns else pd.Series(False, index=scored.index)

    # Pre-filter: keep only targets whose current slot score beats an incumbent somewhere.
    incumbent = {i: analysis.best_11[i]["score"] for i in range(len(analysis.formation))}
    keep: dict[int, list[tuple[float, int]]] = {i: [] for i in incumbent}
    for slot_idx, slot in enumerate(analysis.formation):
        eff = scored[slot["role"]] * fam[slot_idx]
        margin = eff - incumbent[slot_idx]
        # Youth can be below the incumbent now but above over the horizon; allow -2 headroom for u23.
        mask = (fam[slot_idx] > 0) & ((margin > -0.5) | ((age_col <= 23) & (margin > -2.5)))
        for idx in scored.index[mask.values]:
            keep[slot_idx].append((float(margin.loc[idx]), idx))
    cand_idx: set = set()
    for lst in keep.values():
        lst.sort(reverse=True)
        cand_idx.update(i for _, i in lst[: MAX_CANDIDATES_PER_SLOT * 4])

    targets = []
    for idx in cand_idx:
        name = str(name_col.loc[idx])
        if name in squad_names:
            continue
        age = int(age_col.loc[idx])
        position = str(pos_col.loc[idx])
        personality = str(pers_col.loc[idx])
        row = scored.loc[idx]
        sig = extract_signals(row)
        vlo, vhi = parse_value_range(val_col.loc[idx])
        not_for_sale = "not for sale" in str(val_col.loc[idx]).lower()
        horizon_by_slot: dict[int, float] = {}
        band_by_slot: dict[int, tuple[float, float]] = {}
        for slot_idx, slot in enumerate(analysis.formation):
            f = float(fam[slot_idx].loc[idx])
            if f <= 0:
                continue
            cur = float(scored.at[idx, slot["role"]]) * f
            lo, mid, hi = project_band(cur, age, personality, position, seasons, None, sig)
            horizon_by_slot[slot_idx] = _horizon({"low": lo, "high": hi}.get(assumption, mid))
            band_by_slot[slot_idx] = (_horizon(lo), _horizon(hi))
        if not horizon_by_slot:
            continue
        best_slot = max(horizon_by_slot, key=horizon_by_slot.get)
        role_id = analysis.formation[best_slot]["role"]
        targets.append({
            "key": f"target:{idx}",
            "idx": int(idx),
            "name": name,
            "club": str(club_col.loc[idx]),
            "age": age,
            "position": position,
            "personality": personality,
            "wage_k": parse_wage(wage_col.loc[idx]),
            "value_str": str(val_col.loc[idx]),
            "value_lo": vlo,
            "value_hi": vhi,
            "not_for_sale": not_for_sale,
            "cost": _value_for_assumption(vlo, vhi, assumption, buying=True),
            "is_squad": False,
            "needs_scouting": bool(ns_col.loc[idx]),
            "score_now": round(float(scored.at[idx, role_id]) * float(fam[best_slot].loc[idx]), 1),
            "score_lo": round(float(scored.at[idx, f"{role_id}_lo"]), 1),
            "score_hi": round(float(scored.at[idx, f"{role_id}_hi"]), 1),
            "horizon_by_slot": horizon_by_slot,
            "band_by_slot": band_by_slot,
            "best_slot": best_slot,
            "potential": compute_potential_band(float(scored.at[idx, role_id]), age, personality, row)["mid"],
            "signals": sig.notes,
        })
    return targets


def _sell_candidates(analysis: SquadAnalysis, locked: set[str], assumption: str) -> list[dict]:
    recs = generate_sell_recommendations(analysis.squad_df, analysis.formation, analysis.profile, analysis=analysis)
    out = []
    for _, r in recs.iterrows():
        if r["Recommendation"] != "Sell" or r["Name"] in locked:
            continue
        p = analysis.by_idx[int(r["_idx"])]
        if p.is_starter:
            continue
        out.append({
            "key": f"squad:{p.idx}",
            "idx": p.idx,
            "name": p.name,
            "age": p.age,
            "role": p.best_role_name,
            "slot": analysis.formation[p.best_slot_idx]["pos"] if p.best_slot_idx is not None else "-",
            "value_lo": p.value_lo,
            "value_hi": p.value_hi,
            "proceeds": _value_for_assumption(p.value_lo, p.value_hi, assumption, buying=False),
            "wage_k": p.wage_k,
            "priority": float(r["Sell Priority"]),
            "reason": str(r["Reason"]),
            "status": str(r["Status"]),
        })
    out.sort(key=lambda s: s["priority"], reverse=True)
    return out


def generate_strategy(
    squad_df: pd.DataFrame,
    targets_df: pd.DataFrame | None,
    formation_slots: list[dict],
    profile: Profile,
    transfer_budget: float | None = None,
    wage_budget: float | None = None,
    seasons: int = 3,
    max_transfers: int = 6,
    locked_players: set[str] | None = None,
    u21_age: int = 21,
    assumption: str = "mid",
    min_gain: float = 0.3,
    min_gain_per_m: float = 0.05,
    exclude_unscouted: bool = False,
    board_sales_percentage: float = 100.0,
    total_wage_budget: float = 0.0,
    board_slider_cap: float = 1.0,
    current_wage_spend: float = 0.0,
    max_transfer_budget: float = 0.0,
    analysis: SquadAnalysis | None = None,
) -> dict:
    """Build the window plan. See module docstring for the algorithm.

    ``transfer_budget`` / ``wage_budget`` of ``None`` mean "not provided": the
    plan is then funded by sales only and the result carries a warning.
    """
    if not formation_slots:
        formation_slots = [{"pos": "GK", "role": "gkd"}]
    locked = set(locked_players or ())
    if analysis is None:
        analysis = SquadAnalysis(squad_df, formation_slots, profile, assumption)

    warnings: list[str] = []
    budget_missing = transfer_budget is None
    if budget_missing:
        warnings.append("No transfer budget entered — the plan can only be funded by sales. Enter your budget in Quick Start for a realistic plan.")
    transfer_budget = transfer_budget or 0.0
    wage_missing = wage_budget is None
    wage_budget = wage_budget if wage_budget is not None else 1e9  # unconstrained when unknown
    if wage_missing:
        warnings.append("No wage headroom entered — wages of incoming players are reported but not constrained.")
    if assumption == "low":
        warnings.append("Conservative assumption: unknown attributes scored at their lower bound, purchases costed at the top of the asking range, sales at the bottom.")
    elif assumption == "high":
        warnings.append("Optimistic assumption: unknown attributes scored at their upper bound, purchases costed at the bottom of the asking range.")

    pool = _Pool(analysis, seasons, assumption)
    squad_names = {p.name for p in analysis.players}
    targets = _build_targets(targets_df, analysis, seasons, assumption, exclude_unscouted, squad_names)
    for t in targets:
        pool.add_target(t)
    sellable = _sell_candidates(analysis, locked, assumption)
    sold: list[dict] = []
    bought: list[dict] = []
    keys = [f"squad:{p.idx}" for p in analysis.players]

    base_quality, base_lineup = pool.lineup(keys)
    current_now = analysis.total_quality()

    budget = transfer_budget
    wage_room = wage_budget
    sales_factor = board_sales_percentage / 100.0
    skipped_low_value: list[dict] = []
    tried: set[str] = set()

    for _ in range(max_transfers):
        cur_quality, cur_lineup = pool.lineup(keys)
        # Evaluate each candidate's marginal gain against the *current* plan lineup.
        options = []
        for t in targets:
            if t["key"] in tried or t["key"] in keys or t["not_for_sale"]:
                continue
            q, lineup = pool.lineup(keys + [t["key"]])
            gain = round(q - cur_quality, 2)
            if gain < min_gain:
                continue
            cost = max(t["cost"], 0.25)  # floor so free transfers are not infinitely good
            options.append({**t, "gain": gain, "gain_per_m": round(gain / cost, 3), "lineup": lineup})
        if not options:
            break
        options.sort(key=lambda o: (o["gain_per_m"], o["gain"]), reverse=True)
        chosen = None
        for o in options:
            if o["gain_per_m"] < min_gain_per_m and o["cost"] > 1.0:
                skipped_low_value.append({"name": o["name"], "cost": o["cost"], "gain": o["gain"], "gain_per_m": o["gain_per_m"]})
                tried.add(o["key"])
                continue
            if o["wage_k"] > wage_room:
                tried.add(o["key"])
                continue
            # Fund with sales if needed.
            shortfall = o["cost"] - budget
            extra_sales: list[dict] = []
            if shortfall > 0:
                for s in sellable:
                    if s in sold or s in extra_sales:
                        continue
                    trial_keys = [k for k in keys if k != s["key"] and k not in {e["key"] for e in extra_sales}] + [o["key"]]
                    _, trial_lineup = pool.lineup(trial_keys)
                    if not pool.cover_ok(trial_keys, trial_lineup):
                        continue
                    extra_sales.append(s)
                    shortfall -= s["proceeds"] * sales_factor
                    if shortfall <= 0:
                        break
                if shortfall > 0:
                    tried.add(o["key"])
                    continue
            chosen = o
            for s in extra_sales:
                sold.append(s)
                keys.remove(s["key"])
                budget += s["proceeds"] * sales_factor
                wage_room += s["wage_k"]
            break
        if chosen is None:
            break
        bought.append(chosen)
        keys.append(chosen["key"])
        budget -= chosen["cost"]
        wage_room -= chosen["wage_k"]
        tried.add(chosen["key"])

    # Optional: also sell clear surplus that didn't need to fund anything (age 29+ declining, stagnant youth)
    for s in sellable:
        if s in sold or len(sold) >= max_transfers:
            continue
        if s["status"] in ("Declining", "Stagnant") or (s["status"] == "Surplus" and s["age"] >= 24):
            trial_keys = [k for k in keys if k != s["key"]]
            _, trial_lineup = pool.lineup(trial_keys)
            if pool.cover_ok(trial_keys, trial_lineup):
                sold.append(s)
                keys.remove(s["key"])
                budget += s["proceeds"] * sales_factor
                wage_room += s["wage_k"]

    new_quality, new_lineup = pool.lineup(keys)

    # Youth: promote / loan from the sell engine, player-specific reasons.
    recs = generate_sell_recommendations(squad_df, formation_slots, profile, analysis=analysis)
    youth_actions = []
    sold_names = {s["name"] for s in sold}
    for _, r in recs.iterrows():
        if r["Name"] in sold_names or r["Name"] in locked:
            continue
        if r["Recommendation"] in ("Loan", "Promote"):
            youth_actions.append({
                "type": r["Recommendation"].lower(),
                "name": r["Name"],
                "age": int(r["Age"]),
                "role": r["Best Formation Role"],
                "slot": r["Best Slot"],
                "reason": r["Reason"],
            })

    actions = []
    for s in sold:
        actions.append({
            "type": "sell", "name": s["name"], "age": s["age"], "role": s["role"], "slot": s["slot"],
            "value": round(s["proceeds"], 2), "value_lo": s["value_lo"], "value_hi": s["value_hi"],
            "value_label": fmt_millions(s["proceeds"]), "wage_saved": s["wage_k"], "wage_label": fmt_wage(s["wage_k"]),
            "reason": s["reason"],
        })
    for b in bought:
        slot_idx = next((i for i, v in new_lineup.items() if v[0] == b["key"]), b["best_slot"])
        slot = analysis.formation[slot_idx]
        incumbent = analysis.best_11[slot_idx]
        band = b["band_by_slot"].get(slot_idx, (0.0, 0.0))
        actions.append({
            "type": "buy", "name": b["name"], "club": b["club"], "age": b["age"], "position": b["position"],
            "role": ROLES[slot["role"]].name, "slot": slot["pos"],
            "value": round(b["cost"], 2), "value_lo": b["value_lo"], "value_hi": b["value_hi"], "value_str": b["value_str"],
            "value_label": fmt_millions(b["cost"]), "wage": b["wage_k"], "wage_label": fmt_wage(b["wage_k"]),
            "quality_gain": b["gain"], "value_for_money": b["gain_per_m"],
            "score_now": b["score_now"], "score_lo": b["score_lo"], "score_hi": b["score_hi"],
            "horizon": b["horizon_by_slot"].get(slot_idx), "horizon_lo": band[0], "horizon_hi": band[1],
            "needs_scouting": b["needs_scouting"],
            "reason": _buy_reason(b, slot, incumbent, slot_idx, seasons),
        })
    promotes = [a for a in youth_actions if a["type"] == "promote"]
    loans = [a for a in youth_actions if a["type"] == "loan"][:8]
    actions.extend(promotes + loans)

    total_sales = sum(s["proceeds"] for s in sold)
    total_buys = sum(b["cost"] for b in bought)
    wage_in = sum(b["wage_k"] for b in bought)
    wage_out = sum(s["wage_k"] for s in sold)
    squad_wage_bill = sum(p.wage_k for p in analysis.players)

    slider = None
    if total_wage_budget > 0 and max_transfer_budget > 0:
        slider = optimize_slider(
            transfer_budget=transfer_budget, total_wage_budget=total_wage_budget, squad_wage_bill=squad_wage_bill,
            sale_proceeds=total_sales * sales_factor, wage_saved_from_sells=wage_out,
            buy_candidates=[{"buy_value": t["cost"], "wage": t["wage_k"], "quality_gain": t.get("gain", 0.0), "name": t["name"]} for t in bought],
            board_slider_cap=board_slider_cap, current_wage_spend=current_wage_spend, max_transfer_budget=max_transfer_budget,
        )

    projected_lineup = []
    for slot_idx in range(len(analysis.formation)):
        key, h = new_lineup[slot_idx]
        p = pool.players.get(key) if key else None
        projected_lineup.append({
            "pos": analysis.formation[slot_idx]["pos"],
            "role_name": ROLES[analysis.formation[slot_idx]["role"]].name,
            "player_name": p["name"] if p else "(no one)",
            "is_new": bool(p and not p["is_squad"]),
            "horizon": round(h, 1) if key else 0.0,
            "was": analysis.best_11[slot_idx]["player_name"],
        })

    return {
        "assumption": assumption,
        "seasons": seasons,
        "warnings": warnings,
        "budget_missing": budget_missing,
        "diagnosis": {
            "squad_size": len(analysis.players),
            "current_quality_now": current_now,
            "current_quality": base_quality,
            "num_starters": sum(1 for b in analysis.best_11 if b["player_idx"] >= 0),
            "weak_slots": [analysis.slot_label(i) for i in analysis.weak_slots()],
            "thin_slots": [analysis.slot_label(i) for i in range(len(analysis.formation)) if analysis.slot_depth_status(i)["status"] != "ok"],
            "squad_wage_bill": round(squad_wage_bill, 1),
        },
        "actions": actions,
        "skipped_low_value": skipped_low_value[:10],
        "financials": {
            "transfer_budget": transfer_budget,
            "wage_budget": None if wage_missing else wage_budget,
            "total_sales": round(total_sales, 2),
            "available_proceeds": round(total_sales * sales_factor, 2),
            "board_sales_percentage": board_sales_percentage,
            "total_buys": round(total_buys, 2),
            "net_spend": round(total_buys - total_sales, 2),
            "remaining_budget": round(budget, 2),
            "wage_in": round(wage_in, 1),
            "wage_out": round(wage_out, 1),
            "wage_change": round(wage_in - wage_out, 1),
            "labels": {
                "total_sales": fmt_millions(total_sales), "total_buys": fmt_millions(total_buys),
                "net_spend": fmt_millions(total_buys - total_sales, signed=True), "remaining_budget": fmt_millions(budget),
                "wage_change": fmt_wage(wage_in - wage_out, signed=True),
            },
        },
        "projected_quality": {
            "current": base_quality,
            "projected": new_quality,
            "improvement": round(new_quality - base_quality, 2),
            "per_m": round((new_quality - base_quality) / max(total_buys - total_sales, 0.25), 3) if total_buys - total_sales > 0 else None,
            "seasons": seasons,
        },
        "projected_lineup": projected_lineup,
        "budget_slider": slider,
        "reasoning": _generate_reasoning(actions, base_quality, new_quality, analysis, skipped_low_value, assumption, seasons),
    }


def _buy_reason(t: dict, slot: dict, incumbent: dict, slot_idx: int, seasons: int) -> str:
    role_name = ROLES[slot["role"]].name
    h = t["horizon_by_slot"].get(slot_idx, 0.0)
    band = t["band_by_slot"].get(slot_idx, (h, h))
    parts = [f"Projected {h:.1f} (range {band[0]:.1f}-{band[1]:.1f}) at {slot['pos']} {role_name} over {seasons} seasons"]
    if incumbent["player_idx"] >= 0:
        parts.append(f"vs {incumbent['player_name']} {incumbent['score']:.1f} today")
    else:
        parts.append("filling an empty slot")
    parts.append(f"+{t['gain']:.2f} lineup quality for {fmt_millions(t['cost'])} ({t['gain_per_m']:.2f}/£M)")
    if t["age"] <= 23:
        parts.append(f"age {t['age']}, potential ~{t['potential']:.1f}")
    if t["signals"]:
        parts.append("; ".join(t["signals"]))
    if t["needs_scouting"]:
        parts.append("scouting incomplete — verify before bidding")
    return "; ".join(parts)


def _generate_reasoning(actions, current_quality, new_quality, analysis: SquadAnalysis, skipped, assumption, seasons) -> str:
    sells = [a for a in actions if a["type"] == "sell"]
    buys = [a for a in actions if a["type"] == "buy"]
    loans = [a for a in actions if a["type"] == "loan"]
    promotes = [a for a in actions if a["type"] == "promote"]
    label = {"low": "conservative", "high": "optimistic"}.get(assumption, "midpoint")
    parts = [
        f"Suggested plan ({label} assumptions, {seasons}-season horizon; greedy value-for-money search, not a guaranteed optimum).",
        f"Best-XI projected quality {current_quality:.1f} → {new_quality:.1f} ({new_quality - current_quality:+.1f}).",
    ]
    weak = analysis.weak_slots(3)
    if weak:
        parts.append("Weakest slots today: " + ", ".join(f"{analysis.formation[i]['pos']} ({analysis.best_11[i]['score']:.1f})" for i in weak) + ".")
    if buys:
        parts.append("Buy " + ", ".join(f"{b['name']} for {b['slot']} ({b['value_label']}, +{b['quality_gain']:.2f})" for b in buys) + ".")
    else:
        parts.append("No purchase clears the value threshold — the current XI is close to the best available in your shortlist.")
    if sells:
        parts.append("Sell " + ", ".join(f"{s['name']} ({s['value_label']})" for s in sells) + " — none are starters and every slot keeps cover.")
    if promotes:
        parts.append("Promote " + ", ".join(p["name"] for p in promotes) + ".")
    if loans:
        parts.append("Loan out " + ", ".join(f"{ln['name']} ({ln['age']})" for ln in loans) + " for minutes.")
    if skipped:
        parts.append("Skipped as poor value: " + ", ".join(f"{s['name']} ({fmt_millions(s['cost'])} for +{s['gain']:.2f})" for s in skipped[:4]) + ".")
    return " ".join(parts)


def evaluate_plan(
    squad_df: pd.DataFrame,
    targets_df: pd.DataFrame | None,
    formation_slots: list[dict],
    profile: Profile,
    buy_names: list[str],
    sell_names: list[str],
    seasons: int = 3,
    assumption: str = "mid",
    analysis: SquadAnalysis | None = None,
) -> dict:
    """Evaluate a user-defined scenario (Plan A / Plan B): specific buys and sells."""
    if analysis is None:
        analysis = SquadAnalysis(squad_df, formation_slots, profile, assumption)
    pool = _Pool(analysis, seasons, assumption)
    keys = [f"squad:{p.idx}" for p in analysis.players]
    base_q, _ = pool.lineup(keys)

    sells = []
    for p in analysis.players:
        if p.name in sell_names:
            keys.remove(f"squad:{p.idx}")
            sells.append({"name": p.name, "value": _value_for_assumption(p.value_lo, p.value_hi, assumption, False), "wage_k": p.wage_k})

    buys = []
    missing = []
    if targets_df is not None and not targets_df.empty and buy_names:
        sub = targets_df[targets_df["Name"].astype(str).isin(buy_names)]
        sub = sub.drop_duplicates(subset=["Name"])
        for t in _build_targets(sub, analysis, seasons, assumption, False, set()) if not sub.empty else []:
            pool.add_target(t)
            keys.append(t["key"])
            buys.append({"name": t["name"], "value": t["cost"], "wage_k": t["wage_k"], "needs_scouting": t["needs_scouting"]})
        found = {b["name"] for b in buys}
        missing = [n for n in buy_names if n not in found]

    new_q, lineup = pool.lineup(keys)
    total_buys = sum(b["value"] for b in buys)
    total_sales = sum(s["value"] for s in sells)
    lineup_rows = []
    for slot_idx in range(len(analysis.formation)):
        key, h = lineup[slot_idx]
        p = pool.players.get(key) if key else None
        lineup_rows.append({
            "pos": analysis.formation[slot_idx]["pos"],
            "role_name": ROLES[analysis.formation[slot_idx]["role"]].name,
            "player_name": p["name"] if p else "(no one)",
            "is_new": bool(p and not p["is_squad"]),
            "horizon": round(h, 1) if key else 0.0,
        })
    return {
        "quality_before": base_q,
        "quality_after": new_q,
        "improvement": round(new_q - base_q, 2),
        "total_buys": round(total_buys, 2),
        "total_sales": round(total_sales, 2),
        "net_spend": round(total_buys - total_sales, 2),
        "net_spend_label": fmt_millions(total_buys - total_sales, signed=True),
        "wage_change": round(sum(b["wage_k"] for b in buys) - sum(s["wage_k"] for s in sells), 1),
        "wage_change_label": fmt_wage(sum(b["wage_k"] for b in buys) - sum(s["wage_k"] for s in sells), signed=True),
        "per_m": round((new_q - base_q) / (total_buys - total_sales), 3) if total_buys - total_sales > 0 else None,
        "lineup": lineup_rows,
        "buys": buys,
        "sells": sells,
        "missing_targets": missing,
        "cover_ok": pool.cover_ok(keys, lineup),
        "unscouted": [b["name"] for b in buys if b["needs_scouting"]],
        "assumption": assumption,
        "seasons": seasons,
    }


__all__ = ["generate_strategy", "evaluate_plan", "REC_ORDER"]
