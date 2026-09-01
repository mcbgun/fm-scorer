"""Weighting profiles that modify role definitions, loaded from JSON.

A profile defines per-role transformations:
  - promote: move an attribute from one tier to the next higher tier
  - add: add an attribute to a tier (if not already present)
  - remove: remove an attribute from a tier

Built-in profiles ship in ``data/profiles.json`` (Default, Gegenpress,
Possession, Low Block Counter, Wing Play). User-defined profiles are stored in
the user data directory and merged on load.
"""

import json
from dataclasses import dataclass

from paths import BUILTIN_DATA_DIR, data_dir
from roles import ALL_ATTRIBUTES, ROLES, RoleDef

TIERS = ("key", "green", "blue")


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    description: str
    # changes: {role_id: {"promote": {attr: from_tier}, "add": {attr: tier}, "remove": {attr: tier}}}
    changes: dict
    custom: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "changes": self.changes, "custom": self.custom}


PROFILES: dict[str, Profile] = {}


def _custom_profiles_file():
    return data_dir() / "custom_profiles.json"


def load_custom_profiles() -> dict[str, dict]:
    f = _custom_profiles_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_custom_profiles(data: dict[str, dict]) -> None:
    _custom_profiles_file().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    reload_profiles()


def validate_profile_dict(pid: str, d: dict) -> list[str]:
    problems = []
    if not pid or not pid.replace("_", "").isalnum() or len(pid) > 24:
        problems.append("Profile id must be 1-24 alphanumeric characters")
    if not str(d.get("name", "")).strip():
        problems.append("Profile name is required")
    changes = d.get("changes", {})
    if not isinstance(changes, dict):
        return problems + ["changes must be an object keyed by role id"]
    for rid, ch in changes.items():
        if not isinstance(ch, dict):
            problems.append(f"changes for {rid} must be an object")
            continue
        for op in ("promote", "add", "remove"):
            for attr, tier in ch.get(op, {}).items():
                if attr not in ALL_ATTRIBUTES:
                    problems.append(f"{rid}: unknown attribute '{attr}'")
                if tier not in TIERS:
                    problems.append(f"{rid}: unknown tier '{tier}' for {attr}")
    return problems


def reload_profiles() -> None:
    builtin = json.loads((BUILTIN_DATA_DIR / "profiles.json").read_text(encoding="utf-8"))
    profs = {pid: Profile(pid, d["name"], d.get("description", ""), d.get("changes", {})) for pid, d in builtin.items()}
    for pid, d in load_custom_profiles().items():
        if validate_profile_dict(pid, d):
            continue
        profs[pid] = Profile(pid, d["name"], d.get("description", ""), d.get("changes", {}), custom=True)
    PROFILES.clear()
    PROFILES.update(profs)


def get_profile(profile_id: str | None) -> Profile:
    return PROFILES.get(profile_id or "default", PROFILES["default"])


def apply_profile(role: RoleDef, profile: Profile) -> RoleDef:
    """Apply a profile's changes to a role definition, returning a new RoleDef."""
    if role.id not in profile.changes:
        return role

    changes = profile.changes[role.id]
    key = list(role.key)
    green = list(role.green)
    blue = list(role.blue)
    tier_map = {"key": key, "green": green, "blue": blue}

    for attr, from_tier in changes.get("promote", {}).items():
        if attr in tier_map[from_tier]:
            tier_map[from_tier].remove(attr)
        target = "green" if from_tier == "blue" else "key"
        if attr not in tier_map[target]:
            tier_map[target].append(attr)

    for attr, from_tier in changes.get("remove", {}).items():
        if attr in tier_map[from_tier]:
            tier_map[from_tier].remove(attr)

    for attr, to_tier in changes.get("add", {}).items():
        if attr not in key and attr not in green and attr not in blue:
            tier_map[to_tier].append(attr)

    return RoleDef(id=role.id, name=role.name, key=tuple(key), green=tuple(green), blue=tuple(blue), custom=role.custom)


def effective_role(role_id: str, profile: Profile) -> RoleDef:
    return apply_profile(ROLES[role_id], profile)


reload_profiles()
