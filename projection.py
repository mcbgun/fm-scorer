"""Multi-season quality projection for squad planning.

Projects player role scores forward N seasons based on:
  - Youth growth (age <= 21): score grows toward potential
  - Prime stability (22-27): minimal change
  - Age decline (28+): gradual decline, steeper for 31+
  - Goalkeepers: peak later, decline later
  - Exported development signals (training rating, playing time,
    determination, injury susceptibility) when the view includes them

Everything here is an *estimate*: growth in FM depends on hidden attributes,
coaching and match minutes we cannot see, so callers should present a band
(``project_band``) rather than a single number.
"""

import pandas as pd

from development import DevSignals, extract_signals
from youth import compute_potential_score, get_personality_multiplier

# Age-based change per season (points per year), approximating FM24 growth curves.
DECLINE_RATES: dict[int, float] = {
    16: +1.5, 17: +1.3, 18: +1.1, 19: +0.9, 20: +0.7,
    21: +0.5, 22: +0.2, 23: +0.1, 24: 0.0,
    25: 0.0, 26: 0.0, 27: -0.1,
    28: -0.3, 29: -0.4, 30: -0.5,
    31: -0.7, 32: -0.8, 33: -1.0,
    34: -1.2, 35: -1.4,
}

GK_DECLINE_RATES: dict[int, float] = {
    16: +1.5, 17: +1.3, 18: +1.1, 19: +0.9, 20: +0.7,
    21: +0.5, 22: +0.3, 23: +0.2, 24: +0.1,
    25: +0.1, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0,
    30: -0.1, 31: -0.2, 32: -0.3, 33: -0.5,
    34: -0.7, 35: -0.9,
}

# Uncertainty grows with horizon and is larger for youngsters.
UNCERTAINTY_PER_SEASON_YOUTH = 0.6
UNCERTAINTY_PER_SEASON_ADULT = 0.25


def _is_goalkeeper(position: str) -> bool:
    return "GK" in str(position).upper()


def get_decline_rate(age: int, position: str = "") -> float:
    rates = GK_DECLINE_RATES if _is_goalkeeper(position) else DECLINE_RATES
    if age in rates:
        return rates[age]
    if age >= 35:
        return -1.0 if _is_goalkeeper(position) else -1.5
    return 0.0


def project_score(
    current_score: float,
    age: int,
    personality: str,
    position: str,
    seasons: int,
    potential_score: float | None = None,
    signals: DevSignals | None = None,
) -> list[float]:
    """Project a player's role score over N seasons.

    Returns a list of length ``seasons + 1`` (index 0 = now).
    """
    if potential_score is None:
        potential_score = compute_potential_score(current_score, age, personality)
    growth_mult = get_personality_multiplier(personality) * (signals.growth_multiplier() if signals else 1.0)
    availability = signals.availability() if signals else 1.0

    scores = [current_score]
    score = current_score
    for s in range(1, seasons + 1):
        rate = get_decline_rate(age + s, position)
        if rate > 0:
            growth = rate * growth_mult
            score = min(score + growth, max(potential_score, score)) if potential_score > 0 else score + growth
        else:
            # Injury-prone players decline a little faster (lost development / fitness).
            score = score + rate * (2.0 - availability)
        score = max(score, 0.0)
        scores.append(round(score, 1))
    return scores


def project_band(
    current_score: float,
    age: int,
    personality: str,
    position: str,
    seasons: int,
    potential_score: float | None = None,
    signals: DevSignals | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """(low, mid, high) projections; the band widens each season."""
    mid = project_score(current_score, age, personality, position, seasons, potential_score, signals)
    per = UNCERTAINTY_PER_SEASON_YOUTH if age <= 22 else UNCERTAINTY_PER_SEASON_ADULT
    lo = [round(max(0.0, m - per * i), 1) for i, m in enumerate(mid)]
    hi = [round(m + per * i, 1) for i, m in enumerate(mid)]
    return lo, mid, hi


def project_squad(squad_df: pd.DataFrame, role_ids: list[str], scored_df: pd.DataFrame, seasons: int = 3) -> pd.DataFrame:
    """Project all squad players' best-role scores over N seasons."""
    name_col = scored_df.get("Name", pd.Series(["?"] * len(scored_df), index=scored_df.index))
    age_col = pd.to_numeric(scored_df.get("Age", pd.Series([99] * len(scored_df), index=scored_df.index)), errors="coerce").fillna(99)
    pos_col = scored_df.get("Position", pd.Series([""] * len(scored_df), index=scored_df.index))
    pers_col = scored_df.get("Personality", pd.Series([""] * len(scored_df), index=scored_df.index))

    rows = []
    valid = [r for r in role_ids if r in scored_df.columns]
    for idx in scored_df.index:
        if not valid:
            continue
        best_role_id = max(valid, key=lambda r: float(scored_df.at[idx, r]))
        best_score = float(scored_df.at[idx, best_role_id])
        age = int(age_col.loc[idx])
        position = str(pos_col.loc[idx])
        personality = str(pers_col.loc[idx])
        signals = extract_signals(scored_df.loc[idx])
        potential = compute_potential_score(best_score, age, personality)
        lo, mid, hi = project_band(best_score, age, personality, position, seasons, potential, signals)
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
            row[f"score_s{s}"] = mid[s]
            row[f"score_s{s}_lo"] = lo[s]
            row[f"score_s{s}_hi"] = hi[s]
        row["avg_projected"] = round(sum(mid) / len(mid), 1)
        rows.append(row)
    return pd.DataFrame(rows)
