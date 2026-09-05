"""A single, decision-first plan joining squad, finance and development work."""

from __future__ import annotations

from typing import Any

import pandas as pd

from money import parse_value_low, parse_value_range, parse_wage

LOAN_FARM_MAX_COST_M = 0.25
LOAN_FARM_MAX_WAGE_K = 2.0


def _number(value: Any) -> float:
    try:
        number = float(pd.to_numeric(value, errors="coerce"))
        return 0.0 if pd.isna(number) else number
    except (TypeError, ValueError):
        return 0.0


def _player_value(row: Any) -> float:
    lo, hi = parse_value_range(str(row.get("Transfer Value", "")))
    return round((lo + hi) / 2, 2)


def _loan_farm_targets(ctx) -> list[dict]:
    df = ctx.targets
    if df is None or df.empty or "Age" not in df.columns:
        return []
    max_cost = LOAN_FARM_MAX_COST_M
    max_wage = LOAN_FARM_MAX_WAGE_K
    rows = []
    for idx, row in df.iterrows():
        age = _number(row.get("Age"))
        potential = _number(row.get("Potential"))
        if age <= 0 or age > 28:
            continue
        cost = parse_value_low(row.get("Asking Price", "")) or parse_value_low(row.get("Transfer Value", ""))
        wage = parse_wage(str(row.get("Wage", "")))
        if cost > max_cost or wage > max_wage:
            continue
        value_lo, value_hi = parse_value_range(row.get("Transfer Value", ""))
        is_free_agent = "free" in str(row.get("Club", "")).lower() or str(row.get("Transfer Value", "")).strip().lower() in {"free", "free agent"}
        if cost == 0 and not is_free_agent:
            continue
        value_per_week = value_lo * 1000 / 192
        wage_share = min(100, int(value_per_week / wage * 100 // 10) * 10) if wage else 100
        wage_income = wage * wage_share / 100
        fee_per_month = max(0.0, (value_per_week - wage) * 4) if wage_share == 100 else 0.0
        net_month = wage_income + fee_per_month - wage
        season_cash = net_month * 10
        resale_margin = max(0.0, value_hi - cost)
        if is_free_agent and value_lo == 0:
            lane = "Free-agent test"
            score = max(0, potential) * 0.2 - wage
        elif net_month > 0:
            lane = "Cash-flow asset"
            score = net_month * 10 + resale_margin * 0.25 + max(0, 24 - age) * 0.02
        else:
            lane = "Development asset"
            score = resale_margin * 0.25 + max(0, 24 - age) * 0.02
        if net_month <= 0 and resale_margin <= 0 and not is_free_agent:
            continue
        rows.append({
            "idx": int(idx),
            "name": str(row.get("Name", "")),
            "age": int(age),
            "position": str(row.get("Position", "")),
            "potential": round(potential, 1),
            "cost": cost,
            "value": value_lo,
            "resale_value": value_hi,
            "wage": wage,
            "wage_share": wage_share,
            "fee_per_month": round(fee_per_month, 2),
            "net_month": round(net_month, 2),
            "season_cash": round(season_cash, 2),
            "resale_margin": round(resale_margin, 2),
            "lane": lane,
            "cash_flow_known": not is_free_agent or value_lo > 0,
            "club": str(row.get("Club", "")),
            "score": round(score, 2),
            "needs_scouting": bool(row.get("Needs Scouting", False)),
        })
    return sorted(rows, key=lambda item: (item["score"], item["net_month"]), reverse=True)[:8]


def build_club_plan(ctx) -> dict | None:
    """Build a compact plan without replacing the detailed analysis pages."""
    analysis = ctx.analysis
    if analysis is None:
        return None

    strategy = ctx.strategy()
    sell = ctx.sell()
    youth = ctx.youth()
    sale_rows = sell[sell["Recommendation"] == "Sell"] if sell is not None and not sell.empty else pd.DataFrame()
    sale_value = round(sum(_player_value(row) for _, row in sale_rows.iterrows()), 2)
    sale_wages = round(sum(parse_wage(str(row.get("Wage", ""))) for _, row in sale_rows.iterrows()), 1)

    youth_rows = []
    if youth is not None and not youth.empty:
        for idx, row in youth.head(12).iterrows():
            readiness = str(row.get("Readiness", ""))
            if "Unlikely" in readiness:
                lane = "Sell"
            elif "loan" in readiness.lower():
                lane = "Loan farm"
            elif "Ready now" in readiness:
                lane = "First team"
            else:
                lane = "Develop"
            youth_rows.append({
                "idx": int(idx),
                "name": str(row.get("Name", "")),
                "age": int(_number(row.get("Age"))),
                "role": str(row.get("Best Role", "")),
                "score": round(_number(row.get("Best Role Score")), 1),
                "potential": round(_number(row.get("Potential Score")), 1),
                "readiness": readiness,
                "lane": lane,
            })

    first_team = []
    if strategy:
        first_team = [
            {"type": a["type"], "name": a["name"], "reason": a.get("reason", ""), "value": a.get("value", 0)}
            for a in strategy.get("actions", [])
            if a["type"] in ("buy", "sell")
        ][:8]

    next_actions = []
    if strategy and strategy.get("actions"):
        for action in strategy["actions"][:5]:
            next_actions.append({
                "title": f"{action['type'].capitalize()} {action['name']}",
                "detail": action.get("reason", "Included in the current window plan."),
                "lane": "First team" if action["type"] == "buy" else "Finance",
            })
    for row in youth_rows:
        if row["lane"] in ("Loan farm", "Sell"):
            next_actions.append({
                "title": f"{row['lane']}: {row['name']}",
                "detail": row["readiness"] or "Move them into the most useful development pathway.",
                "lane": "Youth",
            })
    if ctx.registration_result() and ctx.registration_result().issues:
        next_actions.insert(0, {
            "title": "Resolve registration constraints",
            "detail": "; ".join(ctx.registration_result().issues[:2]),
            "lane": "Squad",
        })
    if not next_actions:
        next_actions.append({
            "title": "Upload transfer targets",
            "detail": "Add a scouting export to turn the weak-slot analysis into a buying plan.",
            "lane": "Recruitment",
        })

    finance = {
        "wage_bill": round(sum(p.wage_k for p in analysis.players), 1),
        "wage_budget": ctx.settings.get("wage_budget"),
        "transfer_budget": ctx.settings.get("transfer_budget"),
        "sale_value": sale_value,
        "sale_wages": sale_wages,
        "strategy": strategy.get("summary", {}) if strategy else {},
    }
    return {
        "finance": finance,
        "next_actions": next_actions[:8],
        "first_team": first_team,
        "loan_farm": _loan_farm_targets(ctx),
        "youth": youth_rows,
        "best_xi": ctx.pitch_lineup(),
        "depth_alerts": [a for a in ctx.dashboard()["depth_alerts"] if a["status"] != "ok"],
        "notes": [
            "Loan-farm candidates are hard-capped at £250K acquisition cost and £2K p/w wages: this lane is for cheap cash-flow assets, not expensive prospects.",
            "The fee estimate uses a value-to-weekly-wage heuristic; verify it with an actual offer before counting income.",
            "Potential, playing time and facilities are uncertain, so development recommendations are bands rather than guarantees.",
        ],
    }
