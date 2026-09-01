"""Session persistence: saves settings and parsed DataFrames to disk.

Stores the last comparison settings (profile, formation, filters, roles)
and the parsed squad/targets DataFrames as pickle files, so the user
doesn't have to re-upload and re-configure everything each session.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_DIR = Path(__file__).parent
SESSION_FILE = BASE_DIR / "session_state.json"
SQUAD_PKL = BASE_DIR / "session_squad.pkl"
TARGETS_PKL = BASE_DIR / "session_targets.pkl"
REGISTRATION_PKL = BASE_DIR / "session_registration.pkl"


def save_session(
    settings: dict,
    squad_df: pd.DataFrame | None = None,
    targets_df: pd.DataFrame | None = None,
    registration_df: pd.DataFrame | None = None,
) -> None:
    """Save session settings and optional DataFrames to disk.

    Args:
        settings: Dict of profile_id, formation, roles, filters, etc.
        squad_df: Parsed squad DataFrame (with derived stats)
        targets_df: Parsed targets DataFrame (with derived stats)
        registration_df: Parsed registration view DataFrame (Inf column)
    """
    settings["saved_at"] = datetime.now().isoformat(timespec="seconds")
    SESSION_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    if squad_df is not None:
        squad_df.to_pickle(SQUAD_PKL)
    if targets_df is not None:
        targets_df.to_pickle(TARGETS_PKL)
    if registration_df is not None:
        registration_df.to_pickle(REGISTRATION_PKL)


def load_session() -> dict | None:
    """Load saved session settings from disk.

    Returns:
        Settings dict or None if no session saved
    """
    if not SESSION_FILE.exists():
        return None
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_squad() -> pd.DataFrame | None:
    """Load saved squad DataFrame from disk."""
    if not SQUAD_PKL.exists():
        return None
    try:
        return pd.read_pickle(SQUAD_PKL)
    except Exception:
        return None


def load_targets() -> pd.DataFrame | None:
    """Load saved targets DataFrame from disk."""
    if not TARGETS_PKL.exists():
        return None
    try:
        return pd.read_pickle(TARGETS_PKL)
    except Exception:
        return None


def load_registration() -> pd.DataFrame | None:
    """Load saved registration DataFrame from disk."""
    if not REGISTRATION_PKL.exists():
        return None
    try:
        return pd.read_pickle(REGISTRATION_PKL)
    except Exception:
        return None


def has_saved_files() -> bool:
    """Check if saved squad + targets DataFrames exist."""
    return SQUAD_PKL.exists() and TARGETS_PKL.exists()


def has_saved_squad() -> bool:
    """Check if saved squad DataFrame exists."""
    return SQUAD_PKL.exists()


def clear_session() -> None:
    """Delete all saved session data."""
    for f in [SESSION_FILE, SQUAD_PKL, TARGETS_PKL, REGISTRATION_PKL]:
        if f.exists():
            f.unlink()
