"""Formation presets: built-in (``data/formations.json``) plus user-saved.

A formation is a list of slots ``{"pos": "AML", "role": "ifs"}``. Slot position
labels are FM-style: position type (GK, D, WB, DM, M, AM, ST) followed by side
letters (L, C, R), e.g. ``DCR``, ``DMC``, ``AML``, ``STC``.
"""

import json
from datetime import datetime

from paths import BUILTIN_DATA_DIR, data_dir
from positions import parse_slot_position
from roles import ROLES

DEFAULT_FORMATION_ID = "4231gp"


def _custom_file():
    return data_dir() / "custom_formations.json"


def _load_custom() -> dict[str, dict]:
    f = _custom_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _builtin() -> dict[str, dict]:
    return json.loads((BUILTIN_DATA_DIR / "formations.json").read_text(encoding="utf-8"))


def list_formations() -> dict[str, dict]:
    out = {}
    for fid, f in _builtin().items():
        out[fid] = {**f, "id": fid, "custom": False}
    for fid, f in _load_custom().items():
        out[fid] = {**f, "id": fid, "custom": True}
    return out


def get_formation(fid: str) -> dict | None:
    return list_formations().get(fid)


def default_formation() -> dict:
    return list_formations()[DEFAULT_FORMATION_ID]


def validate_slots(slots) -> list[str]:
    problems = []
    if not isinstance(slots, list) or not slots:
        return ["Formation needs at least one slot"]
    if len(slots) > 11:
        problems.append("Formation cannot have more than 11 slots")
    for i, s in enumerate(slots):
        if not isinstance(s, dict) or "pos" not in s or "role" not in s:
            problems.append(f"Slot {i + 1}: needs 'pos' and 'role'")
            continue
        if s["role"] not in ROLES:
            problems.append(f"Slot {i + 1}: unknown role '{s['role']}'")
        ptype, _ = parse_slot_position(str(s["pos"]))
        if ptype is None:
            problems.append(f"Slot {i + 1}: unrecognised position label '{s['pos']}' (use e.g. GK, DR, DCL, DMC, MCR, AML, STC)")
    return problems


def save_formation(fid: str, name: str, slots: list[dict], profile: str | None = None) -> dict:
    fid = fid.strip().lower().replace(" ", "_")
    if not fid or not fid.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Formation id must be alphanumeric")
    problems = validate_slots(slots)
    if problems:
        raise ValueError("; ".join(problems))
    data = _load_custom()
    data[fid] = {
        "name": name.strip() or fid,
        "slots": [{"pos": str(s["pos"]).upper(), "role": s["role"]} for s in slots],
        "profile": profile or "default",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    _custom_file().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return {**data[fid], "id": fid, "custom": True}


def delete_formation(fid: str) -> bool:
    data = _load_custom()
    if fid not in data:
        return False
    del data[fid]
    _custom_file().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return True


def role_slot_counts(slots: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in slots:
        counts[s["role"]] = counts.get(s["role"], 0) + 1
    return counts


def unique_roles(slots: list[dict]) -> list[str]:
    seen: list[str] = []
    for s in slots:
        if s["role"] not in seen:
            seen.append(s["role"])
    return seen
