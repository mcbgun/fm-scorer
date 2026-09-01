"""Squad registration (Premier Division style 25-man rules).

Sources of truth:
  * HGN  - the ``Home-Grown Status`` column. ("Trained at club" also implies
           home-grown *nation*.)
  * U21  - derived from ``DoB`` and the detected season. Premier Division
           rule: exempt if born on/after 1 January of (season start year - 21),
           i.e. aged under 21 (20 or less) on 1 January of the season's start year.
           Verified against FM's own U21 icons in a real registration export.
  * Inf  - FM's status-icon column. It contains *many* different icons (Wnt,
           Inj, Spt, Int, U21, HGN, PR...). We read it only as a list of status
           flags for display; it is never the primary source for HGN/U21.
"""

import unicodedata
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from money import parse_value_range, parse_wage
from season import SeasonInfo, detect_season, parse_dob
from squad_model import SquadAnalysis

HGN_STATUSES = (
    "Trained at club (0-21)",
    "Trained in nation (0-21)",
    "Trained in nation (15-21)",
)
CLUB_TRAINED_STATUSES = ("Trained at club (0-21)",)

INF_STATUS_LABELS = {
    "U21": "Under-21 (exempt)",
    "HGN": "Home-grown (nation)",
    "HGC": "Home-grown (club)",
    "Wnt": "Wanted by other clubs",
    "Inj": "Injured",
    "Spt": "Suspended",
    "Int": "Away on international duty",
    "Unh": "Unhappy",
    "Lst": "Transfer listed",
    "Loa": "Loan listed",
    "PR": "Press / media",
    "Yth": "Youth contract",
    "Ctr": "Contract expiring",
    "Trn": "Training",
    "Fat": "Fatigued",
    "Rec": "Recovering",
}


def is_hgn(hg_status) -> bool:
    return str(hg_status).strip() in HGN_STATUSES


def is_club_trained(hg_status) -> bool:
    return str(hg_status).strip() in CLUB_TRAINED_STATUSES


def is_u21_exempt(dob_str, season_start_year: int | None, u21_age: int = 20) -> bool | None:
    """True if the player is ``u21_age`` or younger on 1 January of the season's start year.

    Returns None when either the DoB or the season is unknown, so callers can
    show "unknown" instead of silently registering a youngster.
    """
    if season_start_year is None:
        return None
    birth, _ = parse_dob(dob_str)
    if birth is None:
        return None
    ref = date(season_start_year, 1, 1)
    age_on_ref = ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    return age_on_ref <= u21_age


def parse_inf_statuses(inf_val) -> list[str]:
    if inf_val is None or (isinstance(inf_val, float) and pd.isna(inf_val)):
        return []
    s = str(inf_val).strip()
    if s in ("", "-", "nan"):
        return []
    tokens = [t.strip() for t in s.replace("/", ",").replace(" ", ",").split(",") if t.strip()]
    return tokens


def inf_labels(tokens: list[str]) -> list[str]:
    return [INF_STATUS_LABELS.get(t, t) for t in tokens]


@dataclass
class RegistrationResult:
    registered: list[dict]
    unregistered: list[dict]
    u21_exempt: list[dict]
    unknown_u21: list[dict]
    hgn_count: int
    non_hgn_count: int
    total_quality: float
    constraints_met: bool
    issues: list[str]
    warnings: list[str]
    max_squad: int
    min_squad: int
    min_hgn: int
    u21_age: int
    season: SeasonInfo
    status_flags: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["season"] = self.season.to_dict()
        return d


def optimize_squad_registration(
    squad_df: pd.DataFrame,
    formation_slots: list[dict],
    profile,
    max_squad: int = 25,
    min_squad: int = 15,
    min_hgn: int = 8,
    u21_age: int = 20,
    season: SeasonInfo | None = None,
    analysis: SquadAnalysis | None = None,
) -> RegistrationResult:
    """Pick the best registered squad that satisfies the HGN quota.

    Player quality = best slot score in the chosen formation (side-aware).
    The search is exact for the HGN constraint: it takes the top ``max_squad``
    by score, then swaps the weakest non-HGN for the strongest unregistered HGN
    until the quota is met — that is the optimal solution for a single count
    constraint, because each swap is the cheapest possible.
    """
    if analysis is None:
        analysis = SquadAnalysis(squad_df, formation_slots, profile, season=season)
    season = season or analysis.season
    s = analysis.scored
    hg_col = s["Home-Grown Status"] if "Home-Grown Status" in s.columns else pd.Series(["-"] * len(s), index=s.index)
    dob_col = s["DoB"] if "DoB" in s.columns else pd.Series([""] * len(s), index=s.index)
    inf_col = s["Inf"] if "Inf" in s.columns else pd.Series([""] * len(s), index=s.index)

    players = []
    warnings: list[str] = []
    status_flags: dict[str, list[str]] = {}
    for p in analysis.players:
        hg = str(hg_col.loc[p.idx])
        u21 = is_u21_exempt(dob_col.loc[p.idx], season.start_year, u21_age)
        tokens = parse_inf_statuses(inf_col.loc[p.idx])
        if tokens:
            status_flags[p.name] = inf_labels(tokens)
        if u21 is None and season.start_year is not None and p.age <= u21_age + 1:
            # No DoB: fall back to exported age (less exact around the 1 Jan cut-off).
            u21 = p.age <= u21_age
            warnings.append(f"{p.name}: no DoB exported, U21 status estimated from age {p.age}")
        players.append({
            "idx": p.idx,
            "name": p.name,
            "age": p.age,
            "score": p.best_score,
            "position": p.position,
            "hg_status": hg,
            "is_hgn": is_hgn(hg),
            "is_club_trained": is_club_trained(hg),
            "is_u21": bool(u21) if u21 is not None else False,
            "u21_known": u21 is not None,
            "statuses": inf_labels(tokens),
            "transfer_value": p.transfer_value,
            "wage": p.wage,
            "value_lo": parse_value_range(p.transfer_value)[0],
            "wage_k": parse_wage(p.wage),
            "best_role": p.best_role_name,
            "is_starter": p.is_starter,
        })

    if season.start_year is None:
        warnings.append("Season could not be detected from the export, so U21 exemptions are unknown — everyone is treated as needing registration.")

    u21_exempt = [p for p in players if p["is_u21"]]
    unknown_u21 = [p for p in players if not p["u21_known"] and p["age"] <= u21_age + 1]
    reg_required = sorted((p for p in players if not p["is_u21"]), key=lambda p: p["score"], reverse=True)

    registered = list(reg_required[:max_squad])
    hgn_pool = [p for p in reg_required if p["is_hgn"]]
    while sum(1 for p in registered if p["is_hgn"]) < min_hgn:
        non_hgn = sorted((p for p in registered if not p["is_hgn"]), key=lambda p: p["score"])
        reg_ids = {p["idx"] for p in registered}
        avail = sorted((p for p in hgn_pool if p["idx"] not in reg_ids), key=lambda p: p["score"], reverse=True)
        if not non_hgn or not avail:
            break
        registered.remove(non_hgn[0])
        registered.append(avail[0])
        registered[-1]["swapped_in_for_hgn"] = True
        non_hgn[0]["dropped_for_hgn"] = True

    registered.sort(key=lambda p: p["score"], reverse=True)
    reg_ids = {p["idx"] for p in registered}
    unregistered = sorted((p for p in reg_required if p["idx"] not in reg_ids), key=lambda p: p["score"], reverse=True)

    hgn_count = sum(1 for p in registered if p["is_hgn"])
    issues = []
    if len(registered) > max_squad:
        issues.append(f"Too many registered players: {len(registered)} > {max_squad}")
    if len(registered) < min_squad:
        issues.append(f"Too few registered players: {len(registered)} < {min_squad}")
    if hgn_count < min_hgn:
        issues.append(f"Not enough home-grown players: {hgn_count} < {min_hgn}. You need {min_hgn - hgn_count} more HGN signing(s) or must register fewer non-HGN players.")
    starters_out = [p["name"] for p in unregistered if p["is_starter"]]
    if starters_out:
        issues.append("Best-XI starters cannot be registered: " + ", ".join(starters_out))

    return RegistrationResult(
        registered=registered,
        unregistered=unregistered,
        u21_exempt=u21_exempt,
        unknown_u21=unknown_u21,
        hgn_count=hgn_count,
        non_hgn_count=len(registered) - hgn_count,
        total_quality=round(sum(p["score"] for p in registered), 1),
        constraints_met=not issues,
        issues=issues,
        warnings=warnings,
        max_squad=max_squad,
        min_squad=min_squad,
        min_hgn=min_hgn,
        u21_age=u21_age,
        season=season,
        status_flags=status_flags,
    )


def merge_registration_view(squad_df: pd.DataFrame, reg_df: pd.DataFrame | None) -> pd.DataFrame:
    """Copy the ``Inf`` status column from a registration export onto the squad
    (matched by normalised name). The squad's own ``Inf`` column is kept where
    the registration view has nothing."""
    if reg_df is None or "Name" not in reg_df.columns or "Inf" not in reg_df.columns:
        return squad_df

    def norm(s) -> str:
        return unicodedata.normalize("NFKC", str(s)).strip().casefold()

    inf_map = {norm(n): ("" if pd.isna(v) else str(v)) for n, v in zip(reg_df["Name"], reg_df["Inf"], strict=True)}
    out = squad_df.copy()
    existing = out["Inf"].astype(str).replace("nan", "") if "Inf" in out.columns else pd.Series([""] * len(out), index=out.index)
    merged = out["Name"].astype(str).map(lambda n: inf_map.get(norm(n), ""))
    out["Inf"] = merged.where(merged != "", existing)
    return out


def registration_season(squad_df: pd.DataFrame, registration_df: pd.DataFrame | None = None) -> SeasonInfo:
    info = detect_season(squad_df)
    if info.start_year is None and registration_df is not None:
        info = detect_season(registration_df)
    return info
