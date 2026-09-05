"""FM24 role definitions (squirrel_plays' scoring system), loaded from JSON.

Each role has three attribute tiers:
  - key:   weight x5 (most important attributes)
  - green: weight x3 (important attributes)
  - blue:  weight x1 (secondary attributes)

Score = (sum(key) * 5 + sum(green) * 3 + sum(blue) * 1) / denominator
where denominator = len(key)*5 + len(green)*3 + len(blue)*1

Built-in roles come from ``data/roles.json``. User-defined roles (created in
the tactic editor) are stored in the user data directory and merged on load;
they can override a built-in role of the same id.
"""

import json
from dataclasses import dataclass

from paths import BUILTIN_DATA_DIR, data_dir

KEY_WEIGHT = 5
GREEN_WEIGHT = 3
BLUE_WEIGHT = 1

ALL_ATTRIBUTES = [
    "Acc", "Pac", "Sta", "Wor", "Str", "Jum", "Agi", "Bal",
    "Ant", "Cnt", "Dec", "Tea", "Pos", "Vis", "Agg", "Bra", "Cmp", "OtB", "Ldr", "Det", "Fla",
    "Cro", "Dri", "Fin", "Fir", "Hea", "Lon", "Mar", "Pas", "Tck", "Tec", "Cor", "Fre", "Pen",
    "Aer", "Cmd", "Han", "Kic", "1v1", "Ref", "TRO", "Thr", "Com", "Ecc", "Pun",
]


@dataclass(frozen=True)
class RoleDef:
    id: str
    name: str
    key: tuple[str, ...]
    green: tuple[str, ...]
    blue: tuple[str, ...]
    custom: bool = False

    @property
    def denominator(self) -> int:
        return len(self.key) * KEY_WEIGHT + len(self.green) * GREEN_WEIGHT + len(self.blue) * BLUE_WEIGHT

    def weights(self) -> dict[str, int]:
        w: dict[str, int] = {}
        for a in self.key:
            w[a] = KEY_WEIGHT
        for a in self.green:
            w[a] = GREEN_WEIGHT
        for a in self.blue:
            w[a] = BLUE_WEIGHT
        return w

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "key": list(self.key), "green": list(self.green), "blue": list(self.blue), "custom": self.custom}


def role_from_dict(rid: str, d: dict, custom: bool = False) -> RoleDef:
    return RoleDef(
        id=rid,
        name=str(d["name"]),
        key=tuple(d.get("key", [])),
        green=tuple(d.get("green", [])),
        blue=tuple(d.get("blue", [])),
        custom=custom,
    )


def validate_role_dict(rid: str, d: dict) -> list[str]:
    """Return a list of validation problems (empty if valid)."""
    problems = []
    if not rid or not rid.replace("_", "").isalnum() or len(rid) > 12:
        problems.append("Role id must be 1-12 alphanumeric characters")
    if not str(d.get("name", "")).strip():
        problems.append("Role name is required")
    seen: set[str] = set()
    for tier in ("key", "green", "blue"):
        attrs = d.get(tier, [])
        if not isinstance(attrs, list):
            problems.append(f"{tier} must be a list")
            continue
        for a in attrs:
            if a not in ALL_ATTRIBUTES:
                problems.append(f"Unknown attribute '{a}' in {tier}")
            if a in seen:
                problems.append(f"Attribute '{a}' appears in more than one tier")
            seen.add(a)
    if not seen:
        problems.append("Role needs at least one attribute")
    return problems


ROLES: dict[str, RoleDef] = {}
ROLE_GROUPS: dict[str, list[str]] = {}
CUSTOM_GROUP = "Custom"


def _custom_roles_file():
    return data_dir() / "custom_roles.json"


def load_custom_roles() -> dict[str, dict]:
    f = _custom_roles_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_custom_roles(data: dict[str, dict]) -> None:
    _custom_roles_file().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    reload_roles()


def reload_roles() -> None:
    """(Re)populate ROLES / ROLE_GROUPS from built-in JSON plus user overrides."""
    builtin = json.loads((BUILTIN_DATA_DIR / "roles.json").read_text(encoding="utf-8"))
    roles = {rid: role_from_dict(rid, d) for rid, d in builtin["roles"].items()}
    groups = {g: list(ids) for g, ids in builtin["groups"].items()}
    custom = load_custom_roles()
    custom_ids = []
    for rid, d in custom.items():
        if validate_role_dict(rid, d):
            continue
        roles[rid] = role_from_dict(rid, d, custom=True)
        if not any(rid in ids for ids in groups.values()):
            custom_ids.append(rid)
    if custom_ids:
        groups[CUSTOM_GROUP] = custom_ids
    ROLES.clear()
    ROLES.update(roles)
    ROLE_GROUPS.clear()
    ROLE_GROUPS.update(groups)


def role_group(role_id: str) -> str:
    for g, ids in ROLE_GROUPS.items():
        if role_id in ids:
            return g
    return CUSTOM_GROUP


reload_roles()
