#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Update stopped: local changes would be overwritten." >&2
    echo "Commit, stash, or remove them, then run ./update.sh again." >&2
    exit 1
fi

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
    echo "Update stopped: repository is in detached HEAD state." >&2
    exit 1
fi

echo "Checking origin/$branch..."
git fetch origin "$branch"
git merge --ff-only "origin/$branch"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Installing pinned dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Updated to $(git rev-parse --short HEAD)."
echo "Start the app with: .venv/bin/uvicorn main:app --reload"
