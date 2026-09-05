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


def _side_letters(sides: str) -> set[str]:
    return {c for c in str(sides).upper() if c in "LCR"}


def role_side_letters(role_id: str) -> set[str]:
    """Union of side letters a role accepts ('' means side-less, e.g. GK/DM)."""
    if role_id not in ROLE_POSITION_MAP:
        return set()
    _, req_sides = ROLE_POSITION_MAP[role_id]
    letters: set[str] = set()
    for s in req_sides:
        letters |= _side_letters(s)
    return letters


def role_position_types(role_id: str) -> set[str]:
    if role_id not in ROLE_POSITION_MAP:
        return set()
    return set(ROLE_POSITION_MAP[role_id][0])


def player_can_play_role(position_str: str, role_id: str) -> bool:
    """Check if a player's position string indicates they can play a role
    anywhere on the pitch (ignoring which side the formation slot is on).
    """
    return position_familiarity(position_str, role_id, None) > 0.0


_SLOT_RE = re.compile(r"^(GK|WB|DM|AM|ST|D|M)\s*([LCR]*)$")

# Custom groupings the user may type as slot labels.
_SLOT_ALIASES = {
    "SW": ("D", "C"),
    "CB": ("D", "C"),
    "CD": ("D", "C"),
    "CM": ("M", "C"),
    "CAM": ("AM", "C"),
    "CDM": ("DM", "C"),
    "LB": ("D", "L"),
    "RB": ("D", "R"),
    "LWB": ("WB", "L"),
    "RWB": ("WB", "R"),
    "LM": ("M", "L"),
    "RM": ("M", "R"),
    "LW": ("AM", "L"),
    "RW": ("AM", "R"),
    "CF": ("ST", "C"),
    "FW": ("ST", "C"),
}


def parse_slot_position(slot_pos: str) -> tuple[str | None, str]:
    """Parse a formation slot label like ``AML`` or ``DCR`` into (type, side).

    Returns (None, "") for unrecognised labels. Side letters are normalised to
    a single letter: L, C, R or "" (unspecified). ``DC``/``DCR``/``DCL`` all
    resolve to a central slot because the R/L suffix on a central slot only
    distinguishes the two centre-backs, not a wide position.
    """
    s = str(slot_pos or "").strip().upper().replace(" ", "").replace("(", "").replace(")", "")
    if not s:
        return None, ""
    if s in _SLOT_ALIASES:
        return _SLOT_ALIASES[s]
    m = _SLOT_RE.match(s)
    if not m:
        return None, ""
    ptype, letters = m.group(1), m.group(2)
    if ptype == "GK":
        return "GK", ""
    if "C" in letters:
        return ptype, "C"
    if "L" in letters and "R" not in letters:
        return ptype, "L"
    if "R" in letters and "L" not in letters:
        return ptype, "R"
    if ptype == "DM":
        return "DM", "C" if not letters else ""
    return ptype, ""


# Position types a player may cover for a role slot type when their exported
# position list is close but not exact (e.g. a D (R) covering a WB slot).
_ADJACENT_TYPES: dict[str, dict[str, float]] = {
    "WB": {"D": 0.92, "M": 0.85},
    "D": {"WB": 0.90},
    "DM": {"M": 0.90, "D": 0.85},
    "M": {"DM": 0.90, "AM": 0.88},
    "AM": {"M": 0.90, "ST": 0.85},
    "ST": {"AM": 0.85},
}


# Familiarity multiplier for a winger/full-back asked to play the opposite flank.
# Set to 0.0 (strict) to require the exact side.
WRONG_FLANK_FAMILIARITY = 0.8


def position_familiarity(
    position_str: str, role_id: str, slot_pos: str | None, wrong_flank: float = WRONG_FLANK_FAMILIARITY
) -> float:
    """Return how well a player's exported positions fit a role in a slot.

    FM's Position column lists the positions a player is Natural or
    Accomplished in, so an exact type+side match is treated as full
    familiarity (1.0). Adjacent position types (a D (R) covering a WB (R)
    slot) are partially familiar. Anything else is 0.0 = cannot play.

    When ``slot_pos`` is None the side of the slot is unknown and any side the
    role accepts is fine (legacy behaviour used for role-only filtering).
    """
    if role_id not in ROLE_POSITION_MAP:
        return 1.0

    req_types, req_sides = ROLE_POSITION_MAP[role_id]
    side_less = req_sides == {""}
    role_sides = role_side_letters(role_id)

    slot_type, slot_side = (None, "")
    if slot_pos:
        slot_type, slot_side = parse_slot_position(slot_pos)

    # Required side letter for this slot. A wide role in a slot whose side is
    # unknown accepts any of the role's sides.
    if slot_side and slot_side in role_sides:
        need_side = slot_side
    elif slot_side == "C" and side_less:
        need_side = ""
    elif slot_side and not side_less and slot_side not in role_sides:
        # e.g. wide role placed in a central slot: use the slot side anyway so
        # the user gets what they configured.
        need_side = slot_side
    else:
        need_side = None

    best = 0.0
    for ptype, psides in parse_position_string(position_str):
        pletters = _side_letters(psides)
        type_factor = 0.0
        if ptype in req_types:
            type_factor = 1.0
        elif slot_type and ptype in _ADJACENT_TYPES.get(slot_type, {}) and slot_type in req_types:
            type_factor = _ADJACENT_TYPES[slot_type][ptype]
        elif not slot_type:
            for rt in req_types:
                type_factor = max(type_factor, _ADJACENT_TYPES.get(rt, {}).get(ptype, 0.0))
        if type_factor == 0.0:
            continue

        if side_less and ptype in ("GK", "DM"):
            best = max(best, type_factor)
            continue
        if side_less:
            # Side-less role (DM) but player type is M/D with sides: need central
            if "C" in pletters or not pletters:
                best = max(best, type_factor)
            continue
        if not pletters:
            continue
        if need_side is None:
            if pletters & role_sides:
                best = max(best, type_factor)
        elif need_side == "":
            best = max(best, type_factor)
        elif need_side in pletters:
            best = max(best, type_factor)
        elif need_side in ("L", "R") and ("L" in pletters or "R" in pletters):
            # Plays the other flank: can swap wings at a familiarity cost.
            best = max(best, type_factor * wrong_flank)
    return round(best, 3)


def player_can_play_slot(position_str: str, slot_pos: str, role_id: str, wrong_flank: float = WRONG_FLANK_FAMILIARITY) -> bool:
    return position_familiarity(position_str, role_id, slot_pos, wrong_flank) > 0.0
