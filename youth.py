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

from roles import ROLES
from profiles import Profile
from scorer import score_all_roles


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


def compute_potential_score(
    role_score: float,
    age: int | float,
    personality: str,
) -> float:
    """Compute an age+personality-weighted potential score.

    Formula: role_score * age_multiplier * personality_multiplier

    A 16-year-old Model Citizen scoring 60 on a role gets:
      60 * 1.40 * 1.30 = 109.2

    A 23-year-old Balanced scoring 60 on the same role gets:
      60 * 0.90 * 1.00 = 54.0

    Args:
        role_score: The base role score from the scoring engine
        age: Player age
        personality: FM24 personality string

    Returns:
        Potential score (float), rounded to 1 decimal
    """
    age_mult = get_age_multiplier(age)
    pers_mult = get_personality_multiplier(personality)
    return round(float(role_score) * age_mult * pers_mult, 1)


def analyze_youth(
    df: pd.DataFrame,
    selected_role_ids: list[str],
    profile: Profile,
    max_age: int = 21,
    squad_benchmarks: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Analyze youth players: potential scores, best roles, gap to first team.

    Args:
        df: DataFrame with all players (will be filtered to youth)
        selected_role_ids: roles to evaluate
        profile: weighting profile to apply
        max_age: maximum age to consider "youth" (default 21)
        squad_benchmarks: optional pre-computed first-team benchmarks per role.
            If provided, gap analysis is included. If None, gaps are omitted.

    Returns:
        DataFrame of youth players with columns:
          - Info columns (Name, Age, Position, Personality, etc.)
          - Top 3 role scores
          - Best Role, Best Role Score
          - Potential Score (age+personality weighted)
          - Gap to First Team (if benchmarks provided)
          - Estimated Seasons to First Team (if benchmarks provided)
    """
    # Filter to youth
    if "Age" not in df.columns:
        return df.iloc[0:0]

    age_col = pd.to_numeric(df["Age"], errors="coerce").fillna(99)
    youth_mask = age_col <= max_age
    youth_df = df[youth_mask].copy()

    if youth_df.empty:
        return youth_df

    # Score all selected roles
    scored = score_all_roles(youth_df, selected_role_ids, profile)

    # Get personality column
    pers_col = scored.get("Personality", pd.Series([""] * len(scored)))
    age_series = pd.to_numeric(scored["Age"], errors="coerce").fillna(99)

    # Compute derived columns
    if selected_role_ids:
        role_cols = scored[selected_role_ids]
        best_role_score = role_cols.max(axis=1).round(1)
        best_role_ids = role_cols.idxmax(axis=1)
        best_role_name = best_role_ids.map(lambda rid: ROLES[rid].name if rid in ROLES else "Unknown")
        potential_scores = pd.Series(
            [compute_potential_score(s, a, p) for s, a, p in zip(best_role_score, age_series, pers_col)],
            index=scored.index,
        )
    else:
        best_role_score = pd.Series([0.0] * len(scored), index=scored.index)
        best_role_ids = pd.Series([""] * len(scored), index=scored.index)
        best_role_name = pd.Series([""] * len(scored), index=scored.index)
        potential_scores = pd.Series([0.0] * len(scored), index=scored.index)

    # Gap analysis against first-team benchmarks
    if squad_benchmarks:
        ft_benchmark = best_role_ids.map(
            lambda rid: squad_benchmarks.get(rid, {}).get("best_score", 0.0)
        )
        ft_player = best_role_ids.map(
            lambda rid: squad_benchmarks.get(rid, {}).get("best_player", "")
        )
        gap = (ft_benchmark - best_role_score).round(1)
        growth_per_season = (potential_scores - best_role_score).clip(lower=1.0)
        est_seasons = (gap / growth_per_season).clip(lower=0).round(1)

    # Assign columns directly (fragmentation warning is cosmetic for small datasets)
    scored["Best Role"] = best_role_name.values
    scored["Best Role ID"] = best_role_ids.values
    scored["Best Role Score"] = best_role_score.values
    scored["Potential Score"] = potential_scores.values

    if squad_benchmarks:
        scored["First Team Benchmark"] = ft_benchmark.values
        scored["First Team Player"] = ft_player.values
        scored["Gap to First Team"] = gap.values
        scored["Est. Seasons to FT"] = est_seasons.values

    # Top 3 roles
    if selected_role_ids:
        for i in range(min(3, len(selected_role_ids))):
            top_n = role_cols.apply(lambda row: row.nlargest(i + 1).iloc[-1], axis=1)
            top_role = role_cols.apply(
                lambda row: row.nlargest(i + 1).idxmin() if i > 0 else row.idxmax(),
                axis=1,
            )
            scored[f"Role {i + 1}"] = top_role.map(lambda rid: ROLES[rid].name if rid in ROLES else "Unknown").values
            scored[f"Role {i + 1} Score"] = top_n.round(1).values

    # Sort by potential score descending
    scored = scored.sort_values("Potential Score", ascending=False)

    return scored


def get_training_focus(
    df: pd.DataFrame,
    role_id: str,
    profile: Profile,
    top_n: int = 3,
) -> list[tuple[str, float, float]]:
    """Recommend training focus attributes for a player's best role.

    Identifies which attributes are lowest relative to the role's key/green/blue weights.

    Args:
        df: Single-row DataFrame for the player
        role_id: The role to optimize for
        profile: weighting profile
        top_n: Number of attributes to recommend

    Returns:
        List of (attribute, current_value, role_weight) tuples, sorted by gap
    """
    if role_id not in ROLES:
        return []

    role = apply_profile(ROLES[role_id], profile)
    weights = {}
    for attr in role.key:
        weights[attr] = 5
    for attr in role.green:
        weights[attr] = 3
    for attr in role.blue:
        weights[attr] = 1

    if df.empty:
        return []

    row = df.iloc[0]
    gaps = []
    for attr, weight in weights.items():
        if attr in row.index:
            val = pd.to_numeric(row[attr], errors="coerce")
            if pd.isna(val):
                continue
            # Gap = (20 - current) * weight — bigger gap + higher weight = more important to train
            gap = (20.0 - float(val)) * weight
            gaps.append((attr, float(val), weight, gap))

    gaps.sort(key=lambda x: x[3], reverse=True)
    return [(attr, val, weight) for attr, val, weight, _ in gaps[:top_n]]


from profiles import apply_profile  # noqa: E402  (circular-safe import at bottom)
