"""Multi-season quality projection for squad planning.

Projects player role scores forward N seasons based on:
  - Youth growth (age <= 21): score grows toward potential
  - Prime stability (22-27): minimal change
  - Age decline (28+): gradual decline, steeper for 31+
  - Goalkeepers: peak later, decline later

Used by the strategy engine to evaluate transfers over multiple seasons,
not just the immediate upgrade.
"""

import pandas as pd

from youth import get_age_multiplier, get_personality_multiplier, compute_potential_score


# Age-based decline rates per season (points lost per year)
# These are approximate, based on FM24 growth curves
DECLINE_RATES: dict[int, float] = {
    # Youth: growth phase (score increases)
    16: +1.5, 17: +1.3, 18: +1.1, 19: +0.9, 20: +0.7,
    21: +0.5, 22: +0.2, 23: +0.1, 24: 0.0,
    # Prime: stable
    25: 0.0, 26: 0.0, 27: -0.1,
    # Decline phase
    28: -0.3, 29: -0.4, 30: -0.5,
    31: -0.7, 32: -0.8, 33: -1.0,
    34: -1.2, 35: -1.4,
}

# Goalkeepers decline later
GK_DECLINE_RATES: dict[int, float] = {
    16: +1.5, 17: +1.3, 18: +1.1, 19: +0.9, 20: +0.7,
    21: +0.5, 22: +0.3, 23: +0.2, 24: +0.1,
    25: +0.1, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0,
    30: -0.1, 31: -0.2, 32: -0.3, 33: -0.5,
    34: -0.7, 35: -0.9,
}


def _is_goalkeeper(position: str) -> bool:
    """Check if a player is a goalkeeper based on position string."""
    return "GK" in str(position).upper()


def get_decline_rate(age: int, position: str = "") -> float:
    """Get the per-season quality change for a player.

    Args:
        age: Current player age
        position: Position string (to detect goalkeepers)

    Returns:
        Points change per season (positive = growth, negative = decline)
    """
    if _is_goalkeeper(position):
        rates = GK_DECLINE_RATES
    else:
        rates = DECLINE_RATES

    if age in rates:
        return rates[age]
    if age >= 35:
        return -1.5 if not _is_goalkeeper(position) else -1.0
    return 0.0


def project_score(
    current_score: float,
    age: int,
    personality: str,
    position: str,
    seasons: int,
    potential_score: float | None = None,
) -> list[float]:
    """Project a player's role score over N seasons.

    Args:
        current_score: Current role score
        age: Current age
        personality: FM24 personality string
        position: Position string
        seasons: Number of seasons to project
        potential_score: Pre-computed potential score (optional)

    Returns:
        List of projected scores, length = seasons + 1
        (index 0 = current, index 1 = next season, etc.)
    """
    if potential_score is None:
        potential_score = compute_potential_score(current_score, age, personality)

    scores = [current_score]
    score = current_score

    for s in range(1, seasons + 1):
        future_age = age + s
        rate = get_decline_rate(future_age, position)

        if rate > 0:
            # Growth phase: move toward potential, but never exceed it
            # Growth is modulated by personality
            pers_mult = get_personality_multiplier(personality)
            growth = rate * pers_mult
            # Can't grow beyond potential score, but never reduce below current
            if potential_score > 0:
                score = min(score + growth, max(potential_score, score))
            else:
                score = score + growth
        else:
            # Decline phase: lose quality
            score = score + rate

        # Floor at 0
        score = max(score, 0.0)
        scores.append(round(score, 1))

    return scores


def project_squad(
    squad_df: pd.DataFrame,
    role_ids: list[str],
    scored_df: pd.DataFrame,
    seasons: int = 3,
) -> pd.DataFrame:
    """Project all squad players' best-role scores over N seasons.

    Args:
        squad_df: Original squad DataFrame with Name, Age, Position, Personality
        role_ids: Formation role IDs to score against
        scored_df: DataFrame from score_all_roles with role score columns
        seasons: Number of seasons to project

    Returns:
        DataFrame with columns:
          - Name, Age, Position, Personality
          - best_role, current_score
          - score_s1, score_s2, ... score_sN (projected scores)
          - avg_projected (average over all seasons including current)
    """
    name_col = scored_df.get("Name", pd.Series(["?"] * len(scored_df)))
    age_col = pd.to_numeric(scored_df.get("Age", pd.Series([99] * len(scored_df))), errors="coerce").fillna(99)
    pos_col = scored_df.get("Position", pd.Series([""] * len(scored_df)))
    pers_col = scored_df.get("Personality", pd.Series([""] * len(scored_df)))

    rows = []
    for idx in scored_df.index:
        # Find best role
        best_role_id = None
        best_score = -999
        for role_id in role_ids:
            if role_id in scored_df.columns:
                s = float(scored_df.at[idx, role_id])
                if s > best_score:
                    best_score = s
                    best_role_id = role_id
        if best_role_id is None:
            continue

        age = int(age_col.loc[idx])
        position = str(pos_col.loc[idx])
        personality = str(pers_col.loc[idx])
        potential = compute_potential_score(best_score, age, personality)

        projections = project_score(
            best_score, age, personality, position, seasons, potential
        )

        row = {
            "Name": str(name_col.loc[idx]),
            "Age": age,
            "Position": position,
            "Personality": personality,
            "best_role": best_role_id,
            "current_score": round(best_score, 1),
            "potential": potential,
        }
        for s in range(1, seasons + 1):
            row[f"score_s{s}"] = projections[s]
        row["avg_projected"] = round(sum(projections) / len(projections), 1)
        rows.append(row)

    return pd.DataFrame(rows)
