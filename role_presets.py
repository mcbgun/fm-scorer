"""Persistent role preset storage.

Saves named role selections to a JSON file so users don't have to re-select
roles every session. Presets are stored in presets.json next to the app.
"""

import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PRESETS_FILE = BASE_DIR / "presets.json"


def _load_all() -> dict:
    """Load all presets from the JSON file.

    Returns:
        Dict mapping preset name to {"roles": [...], "created": str}
    """
    if not PRESETS_FILE.exists():
        return {}
    try:
        return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict) -> None:
    """Write all presets to the JSON file."""
    PRESETS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_preset(name: str, role_ids: list[str], formation: list[dict] | None = None) -> dict:
    """Save a named role preset with optional formation.

    Args:
        name: Preset name (e.g. "4-2-4 Gegenpress")
        role_ids: List of role IDs to save
        formation: Optional list of {"pos": "GK", "role": "sks"} slot dicts

    Returns:
        The saved preset dict
    """
    name = name.strip()
    if not name:
        raise ValueError("Preset name cannot be empty")
    if not role_ids:
        raise ValueError("Cannot save a preset with no roles")

    data = _load_all()
    preset = {"roles": role_ids, "created": datetime.now().isoformat(timespec="seconds")}
    if formation:
        preset["formation"] = formation
    data[name] = preset
    _save_all(data)
    return preset


def load_preset(name: str) -> list[str]:
    """Load a named role preset.

    Args:
        name: Preset name

    Returns:
        List of role IDs, or empty list if not found
    """
    data = _load_all()
    preset = data.get(name)
    if preset is None:
        return []
    return preset.get("roles", [])


def delete_preset(name: str) -> bool:
    """Delete a named role preset.

    Args:
        name: Preset name

    Returns:
        True if deleted, False if not found
    """
    data = _load_all()
    if name not in data:
        return False
    del data[name]
    _save_all(data)
    return True


def list_presets() -> dict[str, dict]:
    """List all saved presets.

    Returns:
        Dict mapping name to {"roles": [...], "created": str}
    """
    return _load_all()
