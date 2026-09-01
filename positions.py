"""Position mapping and parsing for FM24 role compatibility checks.

FM24 Position column format examples:
  "GK"
  "D (C), DM, M (C)"
  "D/WB/M/AM (RL)"
  "M (L), AM (RLC), ST (C)"

Each comma-separated entry can have multiple position types separated by "/"
sharing the same side designation in parentheses.
"""

import re


# Role -> (set of position types, set of acceptable sides)
# A player matches if they have any position_type in the set AND the sides overlap
ROLE_POSITION_MAP: dict[str, tuple[set[str], set[str]]] = {
    # Goalkeepers
    "gkd": ({"GK"}, {""}),
    "skd": ({"GK"}, {""}),
    "sks": ({"GK"}, {""}),
    "ska": ({"GK"}, {""}),
    # Central defenders — need D with C in sides
    "bpdd": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "bpds": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "bpdc": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "cdd": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "cds": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "cdc": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "ncbd": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "ncbs": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "ncbc": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "ld": ({"D"}, {"C", "LC", "RC", "RLC"}),
    "ls": ({"D"}, {"C", "LC", "RC", "RLC"}),
    # Wide centre backs — D with L or R
    "wcbd": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wcbs": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wcba": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    # Full backs — D with L or R (no C-only)
    "fbd": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "fbs": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "fba": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "nfbd": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "ifbd": ({"D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    # Wing backs — WB or D with L or R
    "wbd": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wbs": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wba": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "cwbs": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "cwba": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "iwbd": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "iwbs": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "iwba": ({"WB", "D"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    # Defensive midfield — DM
    "ad": ({"DM"}, {""}),
    "hbd": ({"DM"}, {""}),
    "dmd": ({"DM"}, {""}),
    "dms": ({"DM"}, {""}),
    "dlpd": ({"DM"}, {""}),
    "dlps": ({"DM"}, {""}),
    "regs": ({"DM"}, {""}),
    "svs": ({"DM"}, {""}),
    "sva": ({"DM"}, {""}),
    # Central midfield — M (C) or DM
    "bwmd": ({"DM", "M"}, {"C", "LC", "RC", "RLC"}),
    "bwms": ({"DM", "M"}, {"C", "LC", "RC", "RLC"}),
    "cmd": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "cms": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "cma": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "b2bs": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "cars": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "rps": ({"M", "AM"}, {"C", "LC", "RC", "RLC"}),
    # Mezzala — central midfielders who drift wide
    "mezs": ({"M"}, {"C", "LC", "RC", "RLC"}),
    "meza": ({"M"}, {"C", "LC", "RC", "RLC"}),
    # Attacking midfield — AM (C)
    "aps": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    "apa": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    "ams": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    "ama": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    "engs": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    "ssa": ({"AM"}, {"C", "LC", "RC", "RLC"}),
    # Wide attackers — AM or M with L or R
    "ifs": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "ifa": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "iws": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "iwa": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "ws": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wa": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wps": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wpa": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wmd": ({"M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wms": ({"M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wma": ({"M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "dwd": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "dws": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "raua": ({"AM", "M"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    # Forwards — ST (C)
    "afa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "cfs": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "cfa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "dlfs": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "dlfa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "f9s": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "pa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "pfd": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "pfs": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "pfa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "tfs": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "tfa": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    "trea": ({"ST"}, {"C", "LC", "RC", "RLC"}),
    # Wide target forward — AM or ST with L or R
    "wtfs": ({"AM", "ST"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
    "wtfa": ({"AM", "ST"}, {"L", "R", "LC", "RC", "RL", "RLC"}),
}


def parse_position_string(pos_str: str) -> list[tuple[str, str]]:
    """Parse an FM24 Position string into (position_type, sides) tuples.

    Args:
        pos_str: Position column value like "D/WB/M/AM (RL), DM, M (C)"

    Returns:
        List of (position_type, sides) tuples, e.g. [("D","RL"), ("WB","RL"), ("M","RL"), ("AM","RL"), ("DM",""), ("M","C")]
    """
    if not pos_str or str(pos_str).strip() in ("", "nan", "NaN"):
        return []

    s = str(pos_str).strip()
    results = []
    # Split by comma to get individual entries
    entries = [e.strip() for e in s.split(",")]
    for entry in entries:
        # Match: "D/WB/M/AM (RL)" or "DM" or "GK" or "ST (C)"
        m = re.match(r"^([\w/]+)\s*(?:\(([^)]+)\))?$", entry)
        if not m:
            continue
        pos_types = m.group(1).split("/")
        sides = m.group(2) if m.group(2) else ""
        for pt in pos_types:
            results.append((pt.strip(), sides.strip()))
    return results


def player_can_play_role(position_str: str, role_id: str) -> bool:
    """Check if a player's position string indicates they can play a role.

    Args:
        position_str: FM24 Position column value
        role_id: Role ID like "cdd", "wba", "afa"

    Returns:
        True if the player can play the role's position
    """
    if role_id not in ROLE_POSITION_MAP:
        return True  # Unknown role: don't filter

    req_types, req_sides = ROLE_POSITION_MAP[role_id]
    player_positions = parse_position_string(position_str)

    for ptype, psides in player_positions:
        if ptype not in req_types:
            continue
        # Check side overlap
        if req_sides == {""}:
            # Role has no side requirement (GK, DM) — player just needs the type
            return True
        if psides == "":
            # Player has the type but no sides specified — shouldn't happen for non-GK/DM
            continue
        # Check if any required side letter is in the player's sides
        # e.g. req_sides={"C","LC","RC","RLC"}, player sides="LC" -> "C" in "LC" and "L" in "LC"
        # Simple approach: check if any character of req side appears in player sides
        for req_side in req_sides:
            # A side like "C" matches if "C" is in the player's side string
            # A side like "LC" matches if both "L" and "C" are in the player's side string
            if all(c in psides for c in req_side):
                return True
    return False
