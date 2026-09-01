"""FastAPI app for FM24 Player Scorer with preset weighting profiles."""

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from roles import ROLES, ROLE_GROUPS
from profiles import PROFILES
from scorer import (
    score_all_roles,
    compute_derived_stats,
    get_best_role_per_player,
    get_squad_benchmarks,
    filter_upgrades,
    get_best_11,
    filter_formation_upgrades,
)
from parser import parse_html_file, REQUIRED_ATTRS, INFO_COLS
from role_presets import save_preset, load_preset, delete_preset, list_presets
from session_store import save_session, load_session, load_squad, load_targets, load_registration, has_saved_files, has_saved_squad
from youth import analyze_youth, get_training_focus
from sell import generate_sell_recommendations, optimize_squad_registration
from strategy import generate_strategy

app = FastAPI(title="FM24 Player Scorer")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Default formation: 4-2-3-1 DM AM Wide Gegenpress
GEGENPRESS_FORMATION = [
    {"pos": "GK", "role": "sks"},
    {"pos": "DR", "role": "wba"},
    {"pos": "DC", "role": "cdd"},
    {"pos": "DC", "role": "cdd"},
    {"pos": "DL", "role": "wba"},
    {"pos": "DMC", "role": "hbd"},
    {"pos": "DMC", "role": "hbd"},
    {"pos": "AML", "role": "ifs"},
    {"pos": "AMC", "role": "ams"},
    {"pos": "AMR", "role": "ifa"},
    {"pos": "ST", "role": "afa"},
]
GEGENPRESS_ROLES = list({slot["role"] for slot in GEGENPRESS_FORMATION})

# In-memory storage for the current session (MVP: single-user)
# Stores per-screen results so navigating between pages doesn't lose data.
# Keys: "compare", "youth", "sell" — each holds the full template context dict.
# Also stores "scored" for CSV export compatibility.
_session_data: dict = {}


def _clear_screen_results():
    """Clear all stored screen results (call when squad data changes)."""
    for key in ["compare", "youth", "sell", "scored"]:
        _session_data.pop(key, None)


def _store_result(screen: str, context: dict, df: pd.DataFrame | None = None):
    """Store a screen's result context for persistence across navigation.

    Args:
        screen: "compare", "youth", or "sell"
        context: Full template context dict (will be merged with base on retrieval)
        df: Optional DataFrame for CSV export
    """
    # Strip the request object (not serializable, not needed for re-render)
    stored = {k: v for k, v in context.items() if k != "request"}
    _session_data[screen] = stored
    if df is not None:
        _session_data["scored"] = df


def _get_result(screen: str) -> dict | None:
    """Get stored result context for a screen, or None if not run yet."""
    return _session_data.get(screen)


def _parse_registration_file(file_bytes: bytes) -> pd.DataFrame | None:
    """Parse a registration view HTML file and return its DataFrame.

    The registration view is a minimal FM export with Inf showing U21/HGN
    icons and player names.
    """
    import unicodedata
    try:
        try:
            reg_tables = pd.read_html(file_bytes, encoding="utf-8")
        except Exception:
            reg_tables = pd.read_html(file_bytes, encoding="latin-1")
        reg_df = reg_tables[0]
        if "Inf" in reg_df.columns and "Name" in reg_df.columns:
            return reg_df
    except Exception:
        pass
    return None


def _merge_registration(squad_df: pd.DataFrame, reg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge registration Inf values into squad DataFrame by player name."""
    import unicodedata
    def _norm(s):
        return unicodedata.normalize("NFKC", str(s)).strip()
    inf_map = {}
    for _, r in reg_df.iterrows():
        inf_map[_norm(r["Name"])] = str(r["Inf"])
    squad_df = squad_df.copy()
    squad_df["Inf"] = squad_df["Name"].astype(str).map(
        lambda n: inf_map.get(_norm(n), "")
    )
    return squad_df


def _base_context(request: Request, error: str | None = None) -> dict:
    """Build a base template context with common keys."""
    ctx = {
        "request": request,
        "role_groups": ROLE_GROUPS,
        "roles": ROLES,
        "profiles": PROFILES,
        "presets": list_presets(),
        "saved": load_session(),
        "has_saved_files": has_saved_files(),
    }
    if error:
        ctx["error"] = error
    return ctx


def _render_table(df: pd.DataFrame, table_id: str) -> str:
    """Render a DataFrame to HTML with XSS-safe escaping of cell values."""
    # Escape text columns (Name, Position, etc.) but keep numeric columns as-is
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].astype(str).apply(
                lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            )
    return safe_df.to_html(
        table_id=table_id,
        index=False,
        classes="table table-striped table-hover",
        escape=False,  # Already escaped above
    )


def _validate_formation(formation_json: str) -> list[dict] | None:
    """Parse and validate a formation JSON string.

    Returns:
        List of {"pos", "role"} dicts, or None if invalid.
    """
    if not formation_json:
        return None
    try:
        slots = json.loads(formation_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(slots, list):
        return None
    valid = []
    for slot in slots:
        if not isinstance(slot, dict):
            return None
        pos = slot.get("pos", "")
        role = slot.get("role", "")
        if not role or role not in ROLES:
            return None
        valid.append({"pos": pos, "role": role})
    return valid


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page: unified upload + role/profile selection."""
    saved = load_session()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "role_groups": ROLE_GROUPS,
            "roles": ROLES,
            "profiles": PROFILES,
            "presets": list_presets(),
            "saved": saved,
            "has_saved_squad": has_saved_squad(),
            "has_saved_files": has_saved_files(),
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    squad_file: UploadFile = File(None),
    targets_file: UploadFile = File(None),
    registration_file: UploadFile = File(None),
    profile_id: str = Form("default"),
    roles: str = Form(""),
    top_n: int = Form(50),
):
    """Unified upload: save squad, targets, and registration files to session,
    then score the squad with the selected profile and roles."""
    saved = load_session() or {}
    selected_roles = [r.strip() for r in roles.split(",") if r.strip() and r.strip() in ROLES] if roles else []
    if not selected_roles:
        selected_roles = list(ROLES.keys())
    profile = PROFILES.get(profile_id, PROFILES["default"])

    squad_df = None
    targets_df = None
    registration_df = None
    errors = []
    new_upload = False

    # Parse squad
    if squad_file is not None and squad_file.filename:
        new_upload = True
        squad_bytes = await squad_file.read()
        try:
            squad_df = parse_html_file(squad_bytes)
        except ValueError as e:
            errors.append(f"Squad: {e}")
    else:
        squad_df = load_squad()
        if squad_df is None:
            errors.append("No squad file uploaded and no saved squad found.")

    # Parse targets (optional)
    if targets_file is not None and targets_file.filename:
        new_upload = True
        targets_bytes = await targets_file.read()
        try:
            targets_df = parse_html_file(targets_bytes)
            targets_df = compute_derived_stats(targets_df)
        except ValueError as e:
            errors.append(f"Targets: {e}")
            targets_df = None
    else:
        targets_df = load_targets()

    # Parse registration (optional)
    if registration_file is not None and registration_file.filename:
        new_upload = True
        reg_bytes = await registration_file.read()
        registration_df = _parse_registration_file(reg_bytes)
        if registration_df is None:
            errors.append("Registration: could not parse file or missing Inf/Name columns.")
    else:
        registration_df = load_registration()

    if errors and squad_df is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": saved,
                "has_saved_squad": has_saved_squad(),
                "has_saved_files": has_saved_files(),
                "error": "; ".join(errors),
            },
        )

    # Merge registration into squad if both available
    if squad_df is not None and registration_df is not None:
        squad_df = _merge_registration(squad_df, registration_df)

    if squad_df is not None:
        squad_df = compute_derived_stats(squad_df)

    # Clear stored screen results on new upload
    if new_upload:
        _clear_screen_results()

    # Save everything to session
    formation_slots = saved.get("formation", [])
    # Auto-set Gegenpress formation if none saved yet
    if not formation_slots:
        formation_slots = GEGENPRESS_FORMATION
        selected_roles = GEGENPRESS_ROLES
    save_session(
        {
            "profile_id": profile_id,
            "roles": selected_roles,
            "formation": formation_slots,
        },
        squad_df=squad_df,
        targets_df=targets_df,
        registration_df=registration_df,
    )

    if squad_df is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": load_session(),
                "has_saved_squad": False,
                "has_saved_files": has_saved_files(),
                "error": "No squad data available. Please upload a squad file.",
            },
        )

    # Score the squad with selected roles
    try:
        scored = score_all_roles(squad_df, selected_roles, profile)
    except KeyError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": load_session(),
                "has_saved_squad": True,
                "has_saved_files": has_saved_files(),
                "error": f"Missing attribute column: {e}. Make sure your FM24 view includes all attributes.",
            },
        )

    scored = get_best_role_per_player(squad_df, selected_roles, profile)
    scored = scored.sort_values("Highest Role Score", ascending=False).head(top_n)

    display_info = [c for c in INFO_COLS if c in scored.columns]
    display_cols = display_info + ["Spd", "Work", "SetP"] + selected_roles + ["Highest Role Score", "Resulting Role"]
    _session_data["scored"] = scored[display_cols]

    table_html = _render_table(scored[display_cols], "results")

    upload_summary = []
    if squad_df is not None:
        upload_summary.append(f"Squad: {len(squad_df)} players")
    if targets_df is not None:
        upload_summary.append(f"Targets: {len(targets_df)} players")
    if registration_df is not None:
        u21_count = (registration_df["Inf"] == "U21").sum()
        hgn_count = (registration_df["Inf"] == "HGN").sum()
        upload_summary.append(f"Registration: {len(registration_df)} rows ({u21_count} U21, {hgn_count} HGN)")

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "table_html": table_html,
            "profile_name": profile.name,
            "profile_desc": profile.description,
            "num_roles": len(selected_roles),
            "num_players": len(scored),
            "selected_role_ids": selected_roles,
            "roles": ROLES,
            "upload_summary": upload_summary,
            "errors": errors,
        },
    )


@app.post("/score", response_class=HTMLResponse)
async def score_players(
    request: Request,
    profile_id: str = Form("default"),
    roles: str = Form(""),
    top_n: int = Form(50),
):
    """Score the saved squad with selected profile + roles (no file upload needed)."""
    selected_roles = [r.strip() for r in roles.split(",") if r.strip() and r.strip() in ROLES] if roles else []
    if not selected_roles:
        selected_roles = list(ROLES.keys())
    profile = PROFILES.get(profile_id, PROFILES["default"])

    df = load_squad()
    if df is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": load_session(),
                "has_saved_squad": False,
                "has_saved_files": False,
                "error": "No saved squad found. Upload files on the main page first.",
            },
        )

    try:
        scored = score_all_roles(df, selected_roles, profile)
    except KeyError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": load_session(),
                "has_saved_squad": True,
                "has_saved_files": has_saved_files(),
                "error": f"Missing attribute column: {e}. Make sure your FM24 view includes all attributes.",
            },
        )

    scored = get_best_role_per_player(df, selected_roles, profile)
    scored = scored.sort_values("Highest Role Score", ascending=False).head(top_n)

    display_info = [c for c in INFO_COLS if c in scored.columns]
    display_cols = display_info + ["Spd", "Work", "SetP"] + selected_roles + ["Highest Role Score", "Resulting Role"]
    _session_data["scored"] = scored[display_cols]

    table_html = _render_table(scored[display_cols], "results")

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "table_html": table_html,
            "profile_name": profile.name,
            "profile_desc": profile.description,
            "num_roles": len(selected_roles),
            "num_players": len(scored),
            "selected_role_ids": selected_roles,
            "roles": ROLES,
        },
    )


@app.get("/export")
async def export_csv():
    """Export the current scored results as CSV."""
    if "scored" not in _session_data:
        return JSONResponse({"error": "No data to export. Score some players first."}, status_code=400)

    df = _session_data["scored"]
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_data = output.getvalue()

    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fm24_scored.csv"},
    )


@app.get("/squad", response_class=HTMLResponse)
async def squad_comparison_page(request: Request):
    """Squad comparison page: upload squad + targets, find upgrades.

    If compare results are stored from a previous run, renders them below the form.
    """
    saved = load_session()
    stored = _get_result("compare")
    return templates.TemplateResponse(
        "squad.html",
        {
            "request": request,
            "role_groups": ROLE_GROUPS,
            "roles": ROLES,
            "profiles": PROFILES,
            "presets": list_presets(),
            "saved": saved,
            "has_saved_files": has_saved_files(),
            "stored_results": stored,
        },
    )


@app.get("/compare", response_class=HTMLResponse)
async def compare_results_page(request: Request):
    """Re-display stored comparison results without re-running."""
    stored = _get_result("compare")
    if not stored:
        return templates.TemplateResponse(
            "comparison.html",
            {"request": request, "table_html": "", "error": "No comparison results stored. Run a comparison first."},
        )
    context = dict(stored)
    context["request"] = request
    return templates.TemplateResponse("comparison.html", context)


@app.post("/compare", response_class=HTMLResponse)
async def compare_squad(
    request: Request,
    squad_file: UploadFile = File(None),
    targets_file: UploadFile = File(None),
    profile_id: str = Form("default"),
    roles: str = Form(""),
    formation: str = Form(""),
    min_margin: float = Form(0.0),
    max_age: int = Form(99),
    max_value: str = Form(""),
    top_n: int = Form(50),
    position_mode: str = Form("can_play"),
    one_player_per_role: bool = Form(True),
):
    """Compare squad vs transfer targets and show only upgrades.

    Uses saved squad/targets if no files uploaded. If a formation JSON is
    provided, uses formation-based logic (Best 11 with 11 slots, same role
    can appear multiple times). Otherwise falls back to role-based comparison.
    """
    selected_roles = [r.strip() for r in roles.split(",") if r.strip() and r.strip() in ROLES] if roles else []
    if not selected_roles:
        selected_roles = list(ROLES.keys())

    # Parse and validate formation if provided
    formation_slots = _validate_formation(formation) if formation else None

    profile = PROFILES.get(profile_id, PROFILES["default"])
    saved = load_session() or {}

    # Load squad: upload takes priority, fall back to saved
    squad_df = None
    targets_df = None
    new_upload = False
    if squad_file is not None and squad_file.filename:
        new_upload = True
        squad_bytes = await squad_file.read()
        try:
            squad_df = parse_html_file(squad_bytes)
        except ValueError as e:
            return templates.TemplateResponse(
                "squad.html",
                {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES, "profiles": PROFILES, "presets": list_presets(), "saved": saved, "has_saved_files": has_saved_files(), "error": str(e)},
            )
    else:
        squad_df = load_squad()
        if squad_df is None:
            return templates.TemplateResponse(
                "squad.html",
                {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES, "profiles": PROFILES, "presets": list_presets(), "saved": saved, "has_saved_files": False, "error": "No squad file uploaded and no saved squad found. Upload files on the main page first."},
            )

    if targets_file is not None and targets_file.filename:
        new_upload = True
        targets_bytes = await targets_file.read()
        try:
            targets_df = parse_html_file(targets_bytes)
        except ValueError as e:
            return templates.TemplateResponse(
                "squad.html",
                {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES, "profiles": PROFILES, "presets": list_presets(), "saved": saved, "has_saved_files": has_saved_files(), "error": str(e)},
            )
    else:
        targets_df = load_targets()
        if targets_df is None:
            return templates.TemplateResponse(
                "squad.html",
                {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES, "profiles": PROFILES, "presets": list_presets(), "saved": saved, "has_saved_files": has_saved_files(), "error": "No targets file uploaded and no saved targets found. Upload files on the main page first."},
            )

    if new_upload:
        _clear_screen_results()

    squad_df = compute_derived_stats(squad_df)
    targets_df = compute_derived_stats(targets_df)

    # Save session (preserve formation if already saved)
    save_session(
        {
            "profile_id": profile_id,
            "roles": selected_roles,
            "formation": formation_slots or saved.get("formation", []),
            "min_margin": min_margin,
            "max_age": max_age,
            "max_value": max_value,
            "position_mode": position_mode,
            "one_player_per_role": one_player_per_role,
            "top_n": top_n,
        },
        squad_df=squad_df,
        targets_df=targets_df,
    )

    # Formation mode: Best 11 with per-slot benchmarks
    if formation_slots:
        try:
            best_11 = get_best_11(squad_df, formation_slots, profile)
            upgrades = filter_formation_upgrades(
                targets_df,
                squad_df,
                formation_slots,
                profile,
                min_margin=min_margin,
                max_age=max_age,
                max_value=max_value,
                position_mode=position_mode,
            )
        except KeyError as e:
            return templates.TemplateResponse(
                "squad.html",
                {
                    "request": request,
                    "role_groups": ROLE_GROUPS,
                    "roles": ROLES,
                    "profiles": PROFILES,
                    "presets": list_presets(),
                    "error": f"Missing attribute column: {e}. Make sure both files use Squirrel_plays' FM24 views.",
                },
            )

        upgrades = upgrades.head(top_n)
        display_info = [c for c in INFO_COLS if c in upgrades.columns]
        display_cols = (
            display_info
            + ["Spd", "Work", "SetP"]
            + ["Upgrade Position", "Upgrade Role", "Target Best Score",
               "Squad Player Beaten", "Upgrade Margin"]
        )
        display_cols = [c for c in display_cols if c in upgrades.columns]

        _session_data["scored"] = upgrades[display_cols]

        table_html = _render_table(upgrades[display_cols], "results")

        context = {
            "request": request,
            "table_html": table_html,
            "profile_name": profile.name,
            "profile_desc": profile.description,
            "num_roles": len(formation_slots),
            "num_upgrades": len(upgrades),
            "num_squad": len(squad_df),
            "num_targets": len(targets_df),
            "best_11": best_11,
            "formation": formation_slots,
            "benchmark_rows": [],
            "selected_role_ids": selected_roles,
            "roles": ROLES,
            "min_margin": min_margin,
            "max_age": max_age,
            "max_value": max_value,
            "position_mode": position_mode,
            "one_player_per_role": one_player_per_role,
            "is_formation_mode": True,
        }
        _store_result("compare", context, upgrades[display_cols])
        return templates.TemplateResponse("comparison.html", context)

    # Role-based mode (original)
    try:
        upgrades = filter_upgrades(
            targets_df,
            squad_df,
            selected_roles,
            profile,
            min_margin=min_margin,
            max_age=max_age,
            max_value=max_value,
            require_strict_upgrade=True,
            position_mode=position_mode,
            one_player_per_role=one_player_per_role,
        )
    except KeyError as e:
        return templates.TemplateResponse(
            "squad.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "error": f"Missing attribute column: {e}. Make sure both files use Squirrel_plays' FM24 views.",
            },
        )

    benchmarks = get_squad_benchmarks(squad_df, selected_roles, profile, one_player_per_role)

    benchmark_rows = []
    for role_id in selected_roles:
        if role_id in benchmarks:
            bm = benchmarks[role_id]
            benchmark_rows.append({
                "role_id": role_id,
                "role_name": ROLES[role_id].name,
                "best_score": bm["best_score"],
                "best_player": bm["best_player"],
                "second_score": bm["second_score"],
            })

    upgrades = upgrades.head(top_n)
    display_info = [c for c in INFO_COLS if c in upgrades.columns]
    display_cols = (
        display_info
        + ["Spd", "Work", "SetP"]
        + selected_roles
        + ["Target Best Score", "Target Best Role", "Squad Best Score", "Squad Best Player", "Upgrade Margin"]
    )
    display_cols = [c for c in display_cols if c in upgrades.columns]

    _session_data["scored"] = upgrades[display_cols]

    table_html = _render_table(upgrades[display_cols], "results")

    context = {
        "request": request,
        "table_html": table_html,
        "profile_name": profile.name,
        "profile_desc": profile.description,
        "num_roles": len(selected_roles),
        "num_upgrades": len(upgrades),
        "num_squad": len(squad_df),
        "num_targets": len(targets_df),
        "best_11": None,
        "benchmark_rows": benchmark_rows,
        "selected_role_ids": selected_roles,
        "roles": ROLES,
        "min_margin": min_margin,
        "max_age": max_age,
        "max_value": max_value,
        "position_mode": position_mode,
        "one_player_per_role": one_player_per_role,
        "is_formation_mode": False,
    }
    _store_result("compare", context, upgrades[display_cols])
    return templates.TemplateResponse("comparison.html", context)


@app.post("/recompare", response_class=HTMLResponse)
async def recompare(
    request: Request,
    profile_id: str = Form("default"),
    roles: str = Form(""),
    formation: str = Form(""),
    min_margin: float = Form(0.0),
    max_age: int = Form(99),
    max_value: str = Form(""),
    top_n: int = Form(50),
    position_mode: str = Form("can_play"),
    one_player_per_role: bool = Form(True),
):
    """Re-run comparison using saved files from last session (no upload needed)."""
    squad_df = load_squad()
    targets_df = load_targets()
    if squad_df is None or targets_df is None:
        return templates.TemplateResponse(
            "squad.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "presets": list_presets(),
                "saved": load_session(),
                "has_saved_files": False,
                "error": "No saved files from last session. Please upload files first.",
            },
        )

    # Same logic as /compare but with pre-loaded DataFrames
    selected_roles = [r.strip() for r in roles.split(",") if r.strip() and r.strip() in ROLES] if roles else []
    if not selected_roles:
        selected_roles = list(ROLES.keys())

    formation_slots: list[dict] | None = None
    if formation:
        try:
            formation_slots = json.loads(formation)
        except json.JSONDecodeError:
            formation_slots = None

    profile = PROFILES.get(profile_id, PROFILES["default"])

    # Save updated session settings
    save_session(
        {
            "profile_id": profile_id,
            "roles": selected_roles,
            "formation": formation_slots or [],
            "min_margin": min_margin,
            "max_age": max_age,
            "max_value": max_value,
            "position_mode": position_mode,
            "one_player_per_role": one_player_per_role,
            "top_n": top_n,
        },
    )

    if formation_slots:
        try:
            best_11 = get_best_11(squad_df, formation_slots, profile)
            upgrades = filter_formation_upgrades(
                targets_df, squad_df, formation_slots, profile,
                min_margin=min_margin, max_age=max_age, max_value=max_value,
                position_mode=position_mode,
            )
        except KeyError as e:
            return templates.TemplateResponse(
                "squad.html",
                {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES,
                 "profiles": PROFILES, "presets": list_presets(),
                 "saved": load_session(), "has_saved_files": True,
                 "error": f"Missing attribute column: {e}"},
            )

        upgrades = upgrades.head(top_n)
        display_info = [c for c in INFO_COLS if c in upgrades.columns]
        display_cols = display_info + ["Spd", "Work", "SetP"] + [
            "Upgrade Position", "Upgrade Role", "Target Best Score",
            "Squad Player Beaten", "Upgrade Margin"]
        display_cols = [c for c in display_cols if c in upgrades.columns]
        _session_data["scored"] = upgrades[display_cols]
        table_html = _render_table(upgrades[display_cols], "results")

        context = {
            "request": request, "table_html": table_html,
            "profile_name": profile.name, "profile_desc": profile.description,
            "num_roles": len(formation_slots), "num_upgrades": len(upgrades),
            "num_squad": len(squad_df), "num_targets": len(targets_df),
            "best_11": best_11, "formation": formation_slots,
            "benchmark_rows": [], "selected_role_ids": selected_roles,
            "roles": ROLES, "min_margin": min_margin, "max_age": max_age,
            "max_value": max_value, "position_mode": position_mode,
            "one_player_per_role": one_player_per_role, "is_formation_mode": True,
        }
        _store_result("compare", context, upgrades[display_cols])
        return templates.TemplateResponse("comparison.html", context)

    # Role-based mode
    try:
        upgrades = filter_upgrades(
            targets_df, squad_df, selected_roles, profile,
            min_margin=min_margin, max_age=max_age, max_value=max_value,
            require_strict_upgrade=True, position_mode=position_mode,
            one_player_per_role=one_player_per_role,
        )
    except KeyError as e:
        return templates.TemplateResponse(
            "squad.html",
            {"request": request, "role_groups": ROLE_GROUPS, "roles": ROLES,
             "profiles": PROFILES, "presets": list_presets(),
             "saved": load_session(), "has_saved_files": True,
             "error": f"Missing attribute column: {e}"},
        )

    benchmarks = get_squad_benchmarks(squad_df, selected_roles, profile, one_player_per_role)
    benchmark_rows = []
    for role_id in selected_roles:
        if role_id in benchmarks:
            bm = benchmarks[role_id]
            benchmark_rows.append({
                "role_id": role_id, "role_name": ROLES[role_id].name,
                "best_score": bm["best_score"], "best_player": bm["best_player"],
                "second_score": bm["second_score"],
            })

    upgrades = upgrades.head(top_n)
    display_info = [c for c in INFO_COLS if c in upgrades.columns]
    display_cols = display_info + ["Spd", "Work", "SetP"] + selected_roles + [
        "Target Best Score", "Target Best Role", "Squad Best Score",
        "Squad Best Player", "Upgrade Margin"]
    display_cols = [c for c in display_cols if c in upgrades.columns]
    _session_data["scored"] = upgrades[display_cols]
    table_html = _render_table(upgrades[display_cols], "results")

    context = {
        "request": request, "table_html": table_html,
        "profile_name": profile.name, "profile_desc": profile.description,
        "num_roles": len(selected_roles), "num_upgrades": len(upgrades),
        "num_squad": len(squad_df), "num_targets": len(targets_df),
        "best_11": None, "benchmark_rows": benchmark_rows,
        "selected_role_ids": selected_roles, "roles": ROLES,
        "min_margin": min_margin, "max_age": max_age, "max_value": max_value,
        "position_mode": position_mode, "one_player_per_role": one_player_per_role,
        "is_formation_mode": False,
    }
    _store_result("compare", context, upgrades[display_cols])
    return templates.TemplateResponse("comparison.html", context)


@app.get("/youth", response_class=HTMLResponse)
async def youth_page(request: Request):
    """Youth development tracking page. Shows stored results if available."""
    saved = load_session()
    has_files = has_saved_files()
    stored = _get_result("youth")
    context = {
        "request": request,
        "role_groups": ROLE_GROUPS,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": has_files,
    }
    if stored:
        context.update(stored)
    return templates.TemplateResponse("youth.html", context)


@app.post("/youth", response_class=HTMLResponse)
async def analyze_youth_players(
    request: Request,
    squad_file: UploadFile = File(None),
    profile_id: str = Form("default"),
    roles: str = Form(""),
    max_age: int = Form(21),
    top_n: int = Form(50),
):
    """Analyze youth players from squad HTML or saved session.

    Roles are determined automatically from the saved first-team formation.
    If no formation is saved, falls back to manually selected roles or all roles.
    Benchmarks are computed from senior players only (age > max_age) so youth
    are compared against the actual first team.
    """
    saved = load_session()

    # Determine roles: formation roles take priority, then manual selection, then all
    formation_slots = saved.get("formation", []) if saved else []
    formation_role_ids = []
    if formation_slots:
        formation_role_ids = list({slot["role"] for slot in formation_slots if slot.get("role") and slot["role"] in ROLES})

    manual_roles = [r.strip() for r in roles.split(",") if r.strip() and r.strip() in ROLES] if roles else []

    if formation_role_ids:
        selected_roles = formation_role_ids
    elif manual_roles:
        selected_roles = manual_roles
    else:
        selected_roles = list(ROLES.keys())

    profile = PROFILES.get(profile_id, PROFILES["default"])

    # Load squad: upload takes priority, fall back to saved
    squad_df = None
    new_upload = False
    if squad_file is not None and squad_file.filename:
        new_upload = True
        squad_bytes = await squad_file.read()
        try:
            squad_df = parse_html_file(squad_bytes)
        except ValueError as e:
            return templates.TemplateResponse(
                "youth.html",
                {
                    "request": request,
                    "role_groups": ROLE_GROUPS,
                    "roles": ROLES,
                    "profiles": PROFILES,
                    "saved": saved,
                    "has_saved_files": has_saved_files(),
                    "error": str(e),
                },
            )
    else:
        squad_df = load_squad()
        if squad_df is None:
            return templates.TemplateResponse(
                "youth.html",
                {
                    "request": request,
                    "role_groups": ROLE_GROUPS,
                    "roles": ROLES,
                    "profiles": PROFILES,
                    "saved": saved,
                    "has_saved_files": False,
                    "error": "No squad file uploaded and no saved session found. Please upload a squad HTML file.",
                },
            )

    # New squad uploaded — clear all stored screen results
    if new_upload:
        _clear_screen_results()

    squad_df = compute_derived_stats(squad_df)

    # Save squad to session if newly uploaded
    if new_upload:
        save_session(saved or {}, squad_df=squad_df)

    # Compute first-team benchmarks using ONLY senior players (age > max_age)
    # so youth players are compared against the actual first team, not themselves
    age_col = pd.to_numeric(squad_df["Age"], errors="coerce").fillna(99) if "Age" in squad_df.columns else pd.Series([99] * len(squad_df))
    senior_df = squad_df[age_col > max_age]
    if senior_df.empty:
        senior_df = squad_df
    benchmarks = get_squad_benchmarks(senior_df, selected_roles, profile, one_player_per_role=True)

    # Analyze youth
    try:
        youth_df = analyze_youth(
            squad_df,
            selected_roles,
            profile,
            max_age=max_age,
            squad_benchmarks=benchmarks,
        )
    except KeyError as e:
        return templates.TemplateResponse(
            "youth.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "saved": load_session(),
                "has_saved_files": has_saved_files(),
                "error": f"Missing attribute column: {e}. Make sure your squad export uses Squirrel_plays' FM24 views.",
            },
        )

    if youth_df.empty:
        return templates.TemplateResponse(
            "youth.html",
            {
                "request": request,
                "role_groups": ROLE_GROUPS,
                "roles": ROLES,
                "profiles": PROFILES,
                "saved": load_session(),
                "has_saved_files": has_saved_files(),
                "error": f"No players found aged {max_age} or younger in the squad.",
            },
        )

    youth_df = youth_df.head(top_n)

    # Build display columns
    display_info = [c for c in INFO_COLS if c in youth_df.columns]
    display_cols = display_info + [
        "Best Role", "Best Role Score", "Potential Score",
        "First Team Benchmark", "First Team Player",
        "Gap to First Team", "Est. Seasons to FT",
    ]
    display_cols = [c for c in display_cols if c in youth_df.columns]

    table_html = _render_table(youth_df[display_cols], "youthResults")

    context = {
        "request": request,
        "table_html": table_html,
        "profile_name": profile.name,
        "profile_desc": profile.description,
        "num_youth": len(youth_df),
        "max_age": max_age,
        "selected_role_ids": selected_roles,
        "formation_slots": formation_slots,
        "formation_role_names": [ROLES[rid].name for rid in formation_role_ids if rid in ROLES],
        "role_groups": ROLE_GROUPS,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": load_session(),
        "has_saved_files": has_saved_files(),
    }
    _store_result("youth", context, youth_df[display_cols])
    return templates.TemplateResponse("youth.html", context)


@app.get("/sell", response_class=HTMLResponse)
async def sell_page(request: Request):
    """Sell recommendation page. Shows stored results if available."""
    saved = load_session()
    has_files = has_saved_files()
    stored = _get_result("sell")
    context = {
        "request": request,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": has_files,
    }
    if stored:
        context.update(stored)
    return templates.TemplateResponse("sell.html", context)


@app.post("/sell", response_class=HTMLResponse)
async def sell_recommendations(
    request: Request,
    squad_file: UploadFile = File(None),
    registration_file: UploadFile = File(None),
    profile_id: str = Form("default"),
    max_youth_age: int = Form(21),
    squad_size_limit: int = Form(25),
):
    """Generate sell/loan/keep recommendations for the entire squad.

    Uses the saved formation to determine which roles matter. Each player
    is evaluated against the formation roles, assigned a squad status, and
    given a recommendation (Sell, Loan, Promote, Keep).
    """
    saved = load_session()
    formation_slots = saved.get("formation", []) if saved else []

    profile = PROFILES.get(profile_id, PROFILES["default"])

    # Load squad
    squad_df = None
    new_upload = False
    if squad_file is not None and squad_file.filename:
        new_upload = True
        squad_bytes = await squad_file.read()
        try:
            squad_df = parse_html_file(squad_bytes)
        except ValueError as e:
            return templates.TemplateResponse(
                "sell.html",
                {
                    "request": request,
                    "roles": ROLES,
                    "profiles": PROFILES,
                    "saved": saved,
                    "has_saved_files": has_saved_files(),
                    "error": str(e),
                },
            )
    else:
        squad_df = load_squad()
        if squad_df is None:
            return templates.TemplateResponse(
                "sell.html",
                {
                    "request": request,
                    "roles": ROLES,
                    "profiles": PROFILES,
                    "saved": saved,
                    "has_saved_files": False,
                    "error": "No squad file uploaded and no saved session found. Please upload a squad HTML file.",
                },
            )

    # New squad uploaded — clear all stored screen results
    if new_upload:
        _clear_screen_results()

    squad_df = compute_derived_stats(squad_df)

    # Merge registration info: uploaded file takes priority, then saved
    reg_merged = False
    reg_df = None
    if registration_file is not None and registration_file.filename:
        reg_bytes = await registration_file.read()
        reg_df = _parse_registration_file(reg_bytes)
        if reg_df is not None:
            save_session(saved or {}, registration_df=reg_df)
    else:
        reg_df = load_registration()

    if reg_df is not None and not reg_df.empty:
        squad_df = _merge_registration(squad_df, reg_df)
        reg_merged = True

    # Save squad to session if newly uploaded
    if new_upload:
        save_session(saved or {}, squad_df=squad_df)

    if not formation_slots:
        return templates.TemplateResponse(
            "sell.html",
            {
                "request": request,
                "roles": ROLES,
                "profiles": PROFILES,
                "saved": saved,
                "has_saved_files": has_saved_files(),
                "error": "No formation saved. Go to the Squad Comparison page to set up your formation first.",
            },
        )

    try:
        recommendations = generate_sell_recommendations(
            squad_df,
            formation_slots,
            profile,
            max_youth_age=max_youth_age,
            squad_size_limit=squad_size_limit,
        )
    except KeyError as e:
        return templates.TemplateResponse(
            "sell.html",
            {
                "request": request,
                "roles": ROLES,
                "profiles": PROFILES,
                "saved": saved,
                "has_saved_files": has_saved_files(),
                "error": f"Missing attribute column: {e}. Make sure your squad export uses Squirrel_plays' FM24 views.",
            },
        )

    if recommendations.empty:
        return templates.TemplateResponse(
            "sell.html",
            {
                "request": request,
                "roles": ROLES,
                "profiles": PROFILES,
                "saved": saved,
                "has_saved_files": has_saved_files(),
                "error": "No players found in the squad file.",
            },
        )

    # Build formation role names for display
    formation_role_ids = list({slot["role"] for slot in formation_slots if slot.get("role")})
    formation_role_names = [ROLES[rid].name for rid in formation_role_ids if rid in ROLES]

    # Run squad registration optimizer
    registration = optimize_squad_registration(
        squad_df,
        formation_slots,
        profile,
        max_squad=squad_size_limit,
        min_hgn=8,
        u21_age=20,
    )

    table_html = _render_table(recommendations, "sellTable")

    # Summary counts
    rec_counts = recommendations["Recommendation"].value_counts().to_dict()

    context = {
        "request": request,
        "table_html": table_html,
        "profile_name": profile.name,
        "profile_desc": profile.description,
        "num_players": len(recommendations),
        "max_youth_age": max_youth_age,
        "squad_size_limit": squad_size_limit,
        "formation_slots": formation_slots,
        "formation_role_names": formation_role_names,
        "rec_counts": rec_counts,
        "registration": registration,
        "reg_merged": reg_merged,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": has_saved_files(),
    }
    _store_result("sell", context, recommendations)
    return templates.TemplateResponse("sell.html", context)


@app.get("/quickstart", response_class=HTMLResponse)
async def quickstart(request: Request):
    """Quick Start: auto-run the strategy engine with sensible defaults.

    Uses the saved squad/targets/formation and the Gegenpress profile.
    Financial inputs come from the saved session (if any) or defaults to 0.
    """
    saved = load_session()
    if not saved or not has_saved_squad():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "role_groups": ROLE_GROUPS,
            "roles": ROLES,
            "profiles": PROFILES,
            "presets": list_presets(),
            "saved": saved,
            "has_saved_squad": False,
            "has_saved_files": False,
            "error": "Upload your 3 FM24 files first, then click Quick Start.",
        })

    formation_slots = saved.get("formation", GEGENPRESS_FORMATION)
    profile = PROFILES.get("gegenpress", PROFILES["default"])
    fin = saved.get("financials", {})

    squad_df = load_squad()
    if squad_df is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "role_groups": ROLE_GROUPS,
            "roles": ROLES,
            "profiles": PROFILES,
            "presets": list_presets(),
            "saved": saved,
            "has_saved_squad": False,
            "has_saved_files": False,
            "error": "No saved squad found. Upload files first.",
        })
    squad_df = compute_derived_stats(squad_df)
    targets_df = load_targets()

    try:
        plan = generate_strategy(
            squad_df,
            targets_df,
            formation_slots,
            profile,
            transfer_budget=fin.get("transfer_budget", 0),
            wage_budget=fin.get("wage_budget", 0),
            seasons=fin.get("seasons", 3),
            max_transfers=fin.get("max_transfers", 10),
            locked_players=set(),
            total_wage_budget=fin.get("total_wage_budget", 0),
            board_slider_cap=max(0.0, min(1.0, fin.get("board_slider_cap", 100) / 100.0)),
            current_wage_spend=fin.get("current_wage_spend", 0),
            max_transfer_budget=fin.get("max_transfer_budget", 0),
            board_sales_percentage=fin.get("board_sales_percentage", 100),
        )
    except KeyError as e:
        return templates.TemplateResponse("strategy.html", {
            "request": request, "roles": ROLES, "profiles": PROFILES,
            "saved": saved, "has_saved_files": True,
            "error": f"Missing attribute column: {e}. Make sure your exports use Squirrel_plays' FM24 views.",
        })

    context = {
        "request": request,
        "plan": plan,
        "profile_name": profile.name,
        "profile_desc": profile.description,
        "has_targets": targets_df is not None and not targets_df.empty,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": True,
        "quickstart": True,
    }
    _store_result("strategy", context, None)
    return templates.TemplateResponse("strategy.html", context)


@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request, new: str = ""):
    """Strategy page — shows the transfer plan form or stored results."""
    saved = load_session()
    has_files = has_saved_files()
    stored = _get_result("strategy") if not new else None
    # Pre-fill financial inputs from saved session
    fin = saved.get("financials", {}) if saved else {}
    context = {
        "request": request,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": has_files,
        "saved_transfer_budget": fin.get("transfer_budget", 0),
        "saved_wage_budget": fin.get("wage_budget", 0),
        "saved_total_wage_budget": fin.get("total_wage_budget", 0),
        "saved_max_transfer_budget": fin.get("max_transfer_budget", 0),
        "saved_current_wage_spend": fin.get("current_wage_spend", 0),
        "saved_board_slider_cap": fin.get("board_slider_cap", 100),
        "saved_board_sales_percentage": fin.get("board_sales_percentage", 100),
        "saved_seasons": fin.get("seasons", 3),
        "saved_max_transfers": fin.get("max_transfers", 10),
        "saved_locked_players": fin.get("locked_players", ""),
    }
    if stored:
        context.update(stored)
    return templates.TemplateResponse("strategy.html", context)


@app.post("/strategy", response_class=HTMLResponse)
async def strategy_generate(
    request: Request,
    squad_file: UploadFile = File(None),
    targets_file: UploadFile = File(None),
    profile_id: str = Form("default"),
    transfer_budget: float = Form(0.0),
    wage_budget: float = Form(0.0),
    total_wage_budget: float = Form(0.0),
    max_transfer_budget: float = Form(0.0),
    current_wage_spend: float = Form(0.0),
    board_slider_cap: float = Form(100.0),
    board_sales_percentage: float = Form(100.0),
    seasons: int = Form(3),
    max_transfers: int = Form(10),
    locked_players: str = Form(""),
):
    """Generate the optimal transfer plan."""
    # Auto-convert raw FM24 values to engine units (£M, £K/week).
    if transfer_budget > 1000:
        transfer_budget /= 1_000_000
    if max_transfer_budget > 1000:
        max_transfer_budget /= 1_000_000
    if wage_budget > 100_000:
        wage_budget /= 1_000
    if total_wage_budget > 100_000:
        total_wage_budget /= 1_000
    if current_wage_spend > 100_000:
        current_wage_spend /= 1_000

    saved = load_session()
    formation_slots = saved.get("formation", []) if saved else []
    profile = PROFILES.get(profile_id, PROFILES["default"])

    # Persist financial inputs for next session
    if saved:
        saved["financials"] = {
            "transfer_budget": transfer_budget,
            "wage_budget": wage_budget,
            "total_wage_budget": total_wage_budget,
            "max_transfer_budget": max_transfer_budget,
            "current_wage_spend": current_wage_spend,
            "board_slider_cap": board_slider_cap,
            "board_sales_percentage": board_sales_percentage,
            "seasons": seasons,
            "max_transfers": max_transfers,
            "locked_players": locked_players,
        }
        save_session(saved)

    # Load squad
    squad_df = None
    if squad_file is not None and squad_file.filename:
        squad_bytes = await squad_file.read()
        try:
            squad_df = parse_html_file(squad_bytes)
        except ValueError as e:
            return templates.TemplateResponse("strategy.html", {"request": request, "roles": ROLES, "profiles": PROFILES, "saved": saved, "has_saved_files": has_saved_files(), "error": str(e)})
    else:
        squad_df = load_squad()
        if squad_df is None:
            return templates.TemplateResponse("strategy.html", {"request": request, "roles": ROLES, "profiles": PROFILES, "saved": saved, "has_saved_files": False, "error": "No squad file uploaded and no saved session found."})

    squad_df = compute_derived_stats(squad_df)

    # Load targets: upload takes priority, fall back to saved
    targets_df = None
    if targets_file is not None and targets_file.filename:
        targets_bytes = await targets_file.read()
        try:
            targets_df = parse_html_file(targets_bytes)
            targets_df = compute_derived_stats(targets_df)
        except ValueError:
            targets_df = None
    else:
        targets_df = load_targets()

    if not formation_slots:
        return templates.TemplateResponse("strategy.html", {"request": request, "roles": ROLES, "profiles": PROFILES, "saved": saved, "has_saved_files": has_saved_files(), "error": "No formation saved. Set up your formation on the Squad Comparison page first."})

    locked = set(name.strip() for name in locked_players.split(",") if name.strip()) if locked_players else set()

    # Convert board_slider_cap from percentage (0-100) to fraction (0-1)
    slider_cap_fraction = max(0.0, min(1.0, board_slider_cap / 100.0))

    try:
        plan = generate_strategy(
            squad_df,
            targets_df,
            formation_slots,
            profile,
            transfer_budget=transfer_budget,
            wage_budget=wage_budget,
            seasons=seasons,
            max_transfers=max_transfers,
            locked_players=locked,
            total_wage_budget=total_wage_budget,
            board_slider_cap=slider_cap_fraction,
            current_wage_spend=current_wage_spend,
            max_transfer_budget=max_transfer_budget,
            board_sales_percentage=board_sales_percentage,
        )
    except KeyError as e:
        return templates.TemplateResponse("strategy.html", {"request": request, "roles": ROLES, "profiles": PROFILES, "saved": saved, "has_saved_files": has_saved_files(), "error": f"Missing attribute column: {e}. Make sure your exports use Squirrel_plays' FM24 views."})

    context = {
        "request": request,
        "plan": plan,
        "profile_name": profile.name,
        "profile_desc": profile.description,
        "transfer_budget": transfer_budget,
        "wage_budget": wage_budget,
        "total_wage_budget": total_wage_budget,
        "max_transfer_budget": max_transfer_budget,
        "current_wage_spend": current_wage_spend,
        "board_slider_cap": board_slider_cap,
        "board_sales_percentage": board_sales_percentage,
        "seasons": seasons,
        "max_transfers": max_transfers,
        "locked_players": locked_players,
        "has_targets": targets_df is not None and not targets_df.empty,
        "roles": ROLES,
        "profiles": PROFILES,
        "saved": saved,
        "has_saved_files": has_saved_files(),
    }
    _store_result("strategy", context, None)
    return templates.TemplateResponse("strategy.html", context)


@app.get("/api/roles")
async def get_roles():
    """API endpoint: list all roles."""
    return {rid: {"name": r.name, "key": list(r.key), "green": list(r.green), "blue": list(r.blue)} for rid, r in ROLES.items()}


@app.get("/api/profiles")
async def get_profiles():
    """API endpoint: list all profiles."""
    return {pid: {"name": p.name, "description": p.description} for pid, p in PROFILES.items()}


@app.get("/api/presets")
async def api_list_presets():
    """API endpoint: list all saved role presets."""
    return list_presets()


def _sanitize_preset_name(name: str) -> str | None:
    """Sanitize a preset name to prevent path traversal."""
    import re
    # Remove any path separators, dots, or special chars
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    if not clean or len(clean) > 100:
        return None
    return clean


@app.post("/api/presets/{name}")
async def api_save_preset(name: str, request: Request):
    """API endpoint: save a role preset.

    Body should be JSON: {"roles": ["sks", "wba", ...]}
    """
    safe_name = _sanitize_preset_name(name)
    if not safe_name:
        return JSONResponse({"error": "Invalid preset name"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    role_ids = body.get("roles", []) if isinstance(body, dict) else []
    if not isinstance(role_ids, list):
        return JSONResponse({"error": "roles must be a list"}, status_code=400)
    # Validate role IDs
    invalid = [r for r in role_ids if r not in ROLES]
    if invalid:
        return JSONResponse({"error": f"Unknown role IDs: {invalid}"}, status_code=400)
    try:
        save_preset(safe_name, role_ids)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "name": safe_name, "roles": role_ids}


@app.delete("/api/presets/{name}")
async def api_delete_preset(name: str):
    """API endpoint: delete a role preset."""
    safe_name = _sanitize_preset_name(name)
    if not safe_name:
        return JSONResponse({"error": "Invalid preset name"}, status_code=400)
    deleted = delete_preset(safe_name)
    if not deleted:
        return JSONResponse({"error": "Preset not found"}, status_code=404)
    return {"ok": True, "deleted": safe_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
