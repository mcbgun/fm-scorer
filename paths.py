"""Filesystem locations used by the app.

Built-in tactic data (roles, profiles, formations) ships in ``data/``.
Mutable user data (SQLite workspaces, custom roles/profiles) lives outside the
repository in ``FM_SCORER_DATA_DIR`` (default ``~/.fm-scorer``).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
BUILTIN_DATA_DIR = BASE_DIR / "data"


def data_dir() -> Path:
    override = os.environ.get("FM_SCORER_DATA_DIR")
    path = Path(override).expanduser() if override else Path.home() / ".fm-scorer"
    path.mkdir(parents=True, exist_ok=True)
    return path
