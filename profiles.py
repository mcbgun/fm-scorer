"""Preset weighting profiles that modify role definitions.

A profile defines per-role transformations:
  - promote: move an attribute from one tier to a higher tier
  - add: add an attribute to a tier (if not already present)
  - remove: remove an attribute from a tier

The "default" profile is a no-op (uses Squirrel's original weights).
The "gegenpress" profile applies changes based on FM-Arena empirical testing
and the AssMan gegenpress attribute floors (Stamina/Work Rate >= 14,
Aggression >= 13, Anticipation >= 13).
"""

from dataclasses import dataclass
from roles import RoleDef


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    description: str
    # changes: {role_id: {"promote": {attr: from_tier}, "add": {attr: tier}, "remove": {attr: tier}}}
    changes: dict


PROFILES: dict[str, Profile] = {
    "default": Profile(
        id="default",
        name="Default (Squirrel)",
        description="Squirrel_plays' original weightings. Key x5, Green x3, Blue x1.",
        changes={},
    ),
    "gegenpress": Profile(
        id="gegenpress",
        name="Gegenpress",
        description=(
            "Elevates Anticipation, Aggression, and Teamwork across all roles. "
            "Based on FM-Arena 52k-match testing (Pace/Acc/Mental group highest impact) "
            "and AssMan gegenpress floors (Sta>=14, Wor>=14, Agg>=13, Ant>=13). "
            "Removes Flair from pressing roles (individualism hurts coordinated press)."
        ),
        changes={
            "sks":  {"promote": {"Ant": "green"}, "add": {"Agg": "blue"}, "remove": {"Vis": "blue"}},
            "skd":  {"promote": {"Ant": "green"}, "add": {"Agg": "blue"}, "remove": {"Vis": "blue"}},
            "ska":  {"promote": {"Ant": "green"}, "add": {"Agg": "blue"}, "remove": {"Vis": "blue"}},
            "wba":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "wbs":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "wbd":  {"add": {"Agg": "blue"}},
            "cwba": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cwbs": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cdd":  {"promote": {"Ant": "blue", "Agg": "blue"}, "remove": {"Str": "green"}},
            "cds":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cdc":  {"promote": {"Ant": "green"}, "add": {"Agg": "blue", "Tea": "blue"}, "remove": {"Hea": "blue"}},
            "ncbd": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "ncbs": {"promote": {"Ant": "blue"}},
            "ncbc": {"add": {"Agg": "blue", "Tea": "blue"}},
            "bpdd": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "bpds": {"promote": {"Ant": "blue"}},
            "bpdc": {"add": {"Agg": "blue", "Tea": "blue"}},
            "wcbd": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "wcbs": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "wcba": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "ld":   {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "ls":   {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "fbd":  {"add": {"Agg": "blue"}},
            "fbs":  {"add": {"Agg": "blue"}},
            "fba":  {"add": {"Agg": "blue"}},
            "nfbd": {"add": {"Agg": "blue"}},
            "ifbd": {"promote": {"Ant": "blue"}},
            "iwbd": {"add": {"Agg": "blue"}},
            "iwbs": {"add": {"Agg": "blue"}},
            "iwba": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "hbd":  {"promote": {"Agg": "blue"}},
            "ad":   {"add": {"Agg": "blue"}},
            "dmd":  {"add": {"Agg": "blue"}},
            "dms":  {"add": {"Agg": "blue"}},
            "bwmd": {"promote": {"Agg": "green"}},
            "bwms": {"promote": {"Agg": "green"}},
            "cmd":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cms":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cma":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "b2bs": {"add": {"Agg": "blue"}},
            "cars": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "dlpd": {"add": {"Agg": "blue"}},
            "dlps": {"add": {"Agg": "blue"}},
            "regs": {"add": {"Agg": "blue"}},
            "rps":  {"add": {"Agg": "blue"}},
            "svs":  {"add": {"Agg": "blue"}},
            "sva":  {"add": {"Agg": "blue"}},
            "ams":  {"add": {"Agg": "blue", "Tea": "blue"}, "remove": {"Cmp": "blue"}},
            "ama":  {"add": {"Agg": "blue", "Tea": "blue"}, "remove": {"Cmp": "blue"}},
            "aps":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Cmp": "green"}},
            "apa":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Cmp": "green"}},
            "engs": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "ssa":  {"add": {"Agg": "blue", "Tea": "blue"}},
            "ifs":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "ifa":  {"add": {"Agg": "blue", "Tea": "blue"}, "remove": {"Fla": "blue"}},
            "iws":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "iwa":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "ws":   {"add": {"Agg": "blue"}},
            "wa":   {"add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "wps":  {"add": {"Agg": "blue"}},
            "wpa":  {"add": {"Agg": "blue"}, "remove": {"Fla": "blue"}},
            "wmd":  {"add": {"Agg": "blue"}},
            "wms":  {"add": {"Agg": "blue"}},
            "wma":  {"add": {"Agg": "blue"}},
            "dwd":  {"promote": {"Agg": "blue"}},
            "dws":  {"promote": {"Agg": "blue"}},
            "raua": {"add": {"Agg": "blue"}},
            "afa":  {"promote": {"Ant": "blue", "Wor": "blue", "Sta": "blue"}, "add": {"Agg": "blue"}, "remove": {"Bal": "blue"}},
            "cfs":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "cfa":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "dlfs": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "dlfa": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "f9s":  {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "pa":   {"add": {"Agg": "blue"}},
            "pfd":  {"promote": {"Agg": "green"}},
            "pfs":  {"promote": {"Agg": "green"}},
            "pfa":  {"promote": {"Agg": "green"}},
            "tfs":  {"add": {"Agg": "blue"}},
            "tfa":  {"add": {"Agg": "blue"}},
            "trea": {"promote": {"Ant": "blue"}, "add": {"Agg": "blue"}},
            "wtfs": {"add": {"Agg": "blue"}},
            "wtfa": {"add": {"Agg": "blue"}},
        },
    ),
}


def apply_profile(role: RoleDef, profile: Profile) -> RoleDef:
    """Apply a profile's changes to a role definition, returning a new RoleDef."""
    if profile.id == "default" or role.id not in profile.changes:
        return role

    changes = profile.changes[role.id]
    key = list(role.key)
    green = list(role.green)
    blue = list(role.blue)

    tier_map = {"key": key, "green": green, "blue": blue}

    # Promote: move attr from its current tier to a higher tier
    for attr, from_tier in changes.get("promote", {}).items():
        if attr in tier_map[from_tier]:
            tier_map[from_tier].remove(attr)
        # Determine target tier: promote blue->green, green->key
        if from_tier == "blue":
            target = "green"
        elif from_tier == "green":
            target = "key"
        else:
            target = "key"
        if attr not in tier_map[target]:
            tier_map[target].append(attr)

    # Remove: remove attr from specified tier
    for attr, from_tier in changes.get("remove", {}).items():
        if attr in tier_map[from_tier]:
            tier_map[from_tier].remove(attr)

    # Add: add attr to specified tier (if not already in any tier)
    for attr, to_tier in changes.get("add", {}).items():
        if attr not in key and attr not in green and attr not in blue:
            tier_map[to_tier].append(attr)

    return RoleDef(
        id=role.id,
        name=role.name,
        key=tuple(key),
        green=tuple(green),
        blue=tuple(blue),
    )
