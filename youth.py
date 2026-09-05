"""Youth development tracking: age-weighted potential, personality growth, gap analysis.

Builds on the existing role scoring system to answer:
  - Which youngsters have the most potential?
  - How far is each youth player from the first team?
  - What should they train?

Key concepts:
  - Age multiplier: younger players have more room to grow, so a 16-year-old
    scoring 60 on a role is more valuable than a 23-year-old scoring 60.
  - Personality multiplier: FM24 personalities affect attribute growth rate.
    Model Citizens and Professionals grow faster than Spirited or Casual types.
  - Gap analysis: compares each youth player's role score against the
    first-team benchmark for that role to estimate readiness.
"""

import pandas as pd

from development import extract_signals, star_potential_ceiling
from profiles import Profile, apply_profile
from roles import ROLES
from scorer import familiarity_matrix, score_all_roles

# Age -> growth multiplier. Younger players have higher ceilings.
# Based on FM24 growth curves: most growth happens 16-19, slows 20-22, stops ~24+.
AGE_MULTIPLIERS: dict[int, float] = {
    15: 1.45, 16: 1.40, 17: 1.30, 18: 1.20, 19: 1.10,
    20: 1.05, 21: 1.00, 22: 0.95, 23: 0.90, 24: 0.85,
}


def get_age_multiplier(age: int | float) -> float:
    """Get the growth multiplier for a given age.

    Args:
        age: Player age

    Returns:
        Multiplier (e.g. 1.40 for age 16, 1.00 for age 21, 0.80 for 25+)
    """
    age_int = int(age) if not pd.isna(age) else 25
    return AGE_MULTIPLIERS.get(age_int, 0.80)


# Personality -> growth multiplier.
# Based on FM24 hidden attribute effects: Professionalism and Ambition drive growth.
# These are the visible personality descriptions that map to those hidden attributes.
PERSONALITY_MULTIPLIERS: dict[str, float] = {
    "Model Citizen": 1.30,
    "Model Professional": 1.25,
    "Perfectionist": 1.20,
    "Professional": 1.15,
    "Resolute": 1.10,
    "Fairly Professional": 1.05,
    "Driven": 1.10,
    "Ambitious": 1.08,
    "Determined": 1.08,
    "Resilient": 1.05,
    "Realist": 1.03,
    "Honest": 1.02,
    "Balanced": 1.00,
    "Spirited": 0.95,
    "Light-hearted": 0.92,
    "Jovial": 0.90,
    "Casual": 0.85,
    "Slack": 0.80,
}


def get_personality_multiplier(personality: str) -> float:
    """Get the growth multiplier for a personality description.

    Args:
        personality: FM24 personality string (e.g. "Model Citizen")

    Returns:
        Multiplier (1.30 for Model Citizen, 1.00 for unknown/Balanced)
    """
    if not personality or pd.isna(personality):
        return 1.00
    s = str(personality).strip()
    # Try exact match first
    if s in PERSONALITY_MULTIPLIERS:
        return PERSONALITY_MULTIPLIERS[s]
    # Try partial match (some exports have extra text)
    # Check longest keys first to avoid over-matching (e.g. "Model Professional"
    # should match before "Professional")
    for key in sorted(PERSONALITY_MULTIPLIERS, key=len, reverse=True):
        if key.lower() in s.lower():
            return PERSONALITY_MULTIPLIERS[key]
    return 1.00


def compute_potential_score(role_score: float, age: int | float, personality: str) -> float:
    """Age + personality weighted potential: role_score * age_mult * pers_mult."""
    return round(float(role_score) * get_age_multiplier(age) * get_personality_multiplier(personality), 1)


def compute_potential_band(role_score: float, age: int | float, personality: str, row: pd.Series | None = None) -> dict:
    """Potential estimate with a low/high band and the signals that drove it.

    The heuristic potential is blended with exported development signals when
    present (training rating, playing time, determination, coach potential
    stars). Returns ``{"mid", "lo", "hi", "notes", "growth_multiplier"}``.
    """
    base = compute_potential_score(role_score, age, personality)
    notes: list[str] = []
    growth = 1.0
    if row is not None:
        sig = extract_signals(row)
        growth = sig.growth_multiplier()
        notes = list(sig.notes)
        ceiling = star_potential_ceiling(role_score, sig.potential_stars)
        if ceiling is not None:
            base = round((base + ceiling) / 2, 1)
    # Only the growth *above* the current score is scaled by the signals.
    headroom = max(0.0, base - float(role_score))
    mid = round(float(role_score) + headroom * growth, 1)
    spread = max(0.5, headroom * 0.35)
    return {"mid": mid, "lo": round(max(float(role_score), mid - spread), 1), "hi": round(mid + spread, 1), "notes": notes, "growth_multiplier": growth}


def analyze_youth(
    df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    max_age: int = 21,
    squad_benchmarks: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Analyse youth players: potential (with band), best roles, gap to first team.

    ``Est. Seasons to FT`` is a rough estimate (growth is not linear in FM) and
    is accompanied by ``Readiness`` labels so the UI can avoid false precision.
    """
    if "Age" not in df.columns:
        return df.iloc[0:0]

    age_col = pd.to_numeric(df["Age"], errors="coerce").fillna(99)
    youth_df = df[age_col <= max_age].copy()
    if youth_df.empty:
        return youth_df

    scored = score_all_roles(youth_df, selected_role_ids, profile, with_bounds=True)
    valid = [r for r in selected_role_ids if r in scored.columns]
    pers_col = scored["Personality"] if "Personality" in scored.columns else pd.Series([""] * len(scored), index=scored.index)
    age_series = pd.to_numeric(scored["Age"], errors="coerce").fillna(99)

    new: dict[str, pd.Series] = {}
    if valid:
        fam = familiarity_matrix(scored, [{"pos": "", "role": r} for r in valid])
        role_cols = pd.DataFrame({r: scored[r].where(fam[i] > 0, scored[r] * 0.5) for i, r in enumerate(valid)})
        best_role_ids = role_cols.idxmax(axis=1)
        best_role_score = pd.Series([scored.at[i, r] for i, r in best_role_ids.items()], index=scored.index).round(1)
        bands = [compute_potential_band(s, a, p, scored.loc[i]) for i, (s, a, p) in zip(scored.index, zip(best_role_score, age_series, pers_col, strict=True), strict=True)]
        new["Best Role"] = best_role_ids.map(lambda rid: ROLES[rid].name if rid in ROLES else "Unknown")
        new["Best Role ID"] = best_role_ids
        new["Best Role Score"] = best_role_score
        new["Score Low"] = pd.Series([scored.at[i, f"{r}_lo"] for i, r in best_role_ids.items()], index=scored.index).round(1)
        new["Score High"] = pd.Series([scored.at[i, f"{r}_hi"] for i, r in best_role_ids.items()], index=scored.index).round(1)
        new["Potential Score"] = pd.Series([b["mid"] for b in bands], index=scored.index)
        new["Potential Low"] = pd.Series([b["lo"] for b in bands], index=scored.index)
        new["Potential High"] = pd.Series([b["hi"] for b in bands], index=scored.index)
        new["Growth Signals"] = pd.Series(["; ".join(b["notes"]) for b in bands], index=scored.index)
        new["Growth Multiplier"] = pd.Series([b["growth_multiplier"] for b in bands], index=scored.index)
        for i in range(min(3, len(valid))):
            top_role = role_cols.apply(lambda row, k=i: row.nlargest(k + 1).index[-1], axis=1)
            new[f"Role {i + 1}"] = top_role.map(lambda rid: ROLES[rid].name if rid in ROLES else "Unknown")
            new[f"Role {i + 1} Score"] = pd.Series([scored.at[j, r] for j, r in top_role.items()], index=scored.index).round(1)
    else:
        best_role_ids = pd.Series([""] * len(scored), index=scored.index)
        best_role_score = pd.Series(0.0, index=scored.index)
        new["Best Role"] = best_role_ids
        new["Best Role ID"] = best_role_ids
        new["Best Role Score"] = best_role_score
        new["Potential Score"] = best_role_score
        new["Potential Low"] = best_role_score
        new["Potential High"] = best_role_score
        new["Growth Signals"] = best_role_ids
        new["Growth Multiplier"] = pd.Series(1.0, index=scored.index)

    if squad_benchmarks:
        ft_benchmark = best_role_ids.map(lambda rid: squad_benchmarks.get(rid, {}).get("best_score", 0.0)).astype(float)
        gap = (ft_benchmark - best_role_score).round(1)
        growth_per_season = ((new["Potential Score"] - best_role_score) / 3.0).clip(lower=0.4)
        est = (gap / growth_per_season).clip(lower=0)
        new["First Team Benchmark"] = ft_benchmark
        new["First Team Player"] = best_role_ids.map(lambda rid: squad_benchmarks.get(rid, {}).get("best_player", ""))
        new["Gap to First Team"] = gap
        new["Est. Seasons to FT"] = est.round(1)
        new["Readiness"] = pd.Series(
            [readiness_label(g, e, a) for g, e, a in zip(gap, est, age_series, strict=True)], index=scored.index
        )

    scored = scored.drop(columns=[c for c in new if c in scored.columns])
    scored = pd.concat([scored, pd.DataFrame(new, index=scored.index)], axis=1)
    return scored.sort_values("Potential Score", ascending=False)


def readiness_label(gap: float, est_seasons: float, age: float) -> str:
    if gap <= 0:
        return "Ready now"
    if est_seasons <= 1.0:
        return "Ready within a season"
    if est_seasons <= 2.5:
        return "1-3 seasons away (loan for minutes)"
    if age >= 20:
        return "Unlikely to reach first team"
    return "Long-term project"


def get_training_focus(df: pd.DataFrame, role_id: str, profile: Profile, top_n: int = 3) -> list[tuple[str, float, float]]:
    """Recommend training focus attributes for a role: biggest weighted gap to 20."""
    if role_id not in ROLES or df.empty:
        return []
    role = apply_profile(ROLES[role_id], profile)
    row = df.iloc[0]
    gaps = []
    for attr, weight in role.weights().items():
        if attr in row.index:
            val = pd.to_numeric(row[attr], errors="coerce")
            if pd.isna(val):
                continue
            gaps.append((attr, float(val), weight, (20.0 - float(val)) * weight))
    gaps.sort(key=lambda x: x[3], reverse=True)
    return [(attr, val, weight) for attr, val, weight, _ in gaps[:top_n]]
