"""FastAPI app for the FM24 squad planner.

Routes are thin: they read the active workspace via ``services.WorkspaceContext``
and render templates. All analysis lives in the domain modules; all mutable state
lives in the SQLite ``Store`` (outside the repository).
"""

import csv
import io
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from formations import delete_formation, list_formations, save_formation, validate_slots
from money import fmt_millions, fmt_wage
from parser import REQUIRED_ATTRS
from profiles import PROFILES, load_custom_profiles, save_custom_profiles, validate_profile_dict
from roles import ROLE_GROUPS, ROLES, load_custom_roles, save_custom_roles, validate_role_dict
from scorer import ASSUMPTIONS
from services import DEFAULT_SETTINGS, WorkspaceContext, pitch_layout
from store import Store

app = FastAPI(title="FM24 Squad Planner")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["millions"] = fmt_millions
templates.env.filters["wage"] = fmt_wage
templates.env.filters["tojson_safe"] = lambda v: json.dumps(v, default=str, ensure_ascii=False)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

store = Store()

ASSUMPTION_LABELS = {"low": "Conservative", "mid": "Midpoint", "high": "Optimistic"}
SLOT_POSITIONS = ["GK", "DR", "DCR", "DC", "DCL", "DL", "WBR", "WBL", "DMR", "DMCR", "DMC", "DMCL", "DML",
                  "MR", "MCR", "MC", "MCL", "ML", "AMR", "AMCR", "AMC", "AMCL", "AML", "STCR", "STC", "STCL"]
UPLOAD_KINDS = ("squad", "targets", "registration")


# ------------------------------------------------------------------ helpers
def ctx_for(request: Request) -> WorkspaceContext:
    return WorkspaceContext(store)


def render(name: str, request: Request, ctx: WorkspaceContext, **extra) -> HTMLResponse:
    context = {
        "hdr": ctx.header(),
        "settings": ctx.settings,
        "profiles": PROFILES,
        "formations": list_formations(),
        "assumptions": ASSUMPTION_LABELS,
        "roles": ROLES,
        "flash": request.query_params.get("flash"),
        "flash_kind": request.query_params.get("kind", "success"),
        **extra,
    }
    return templates.TemplateResponse(request, name, context)


def redirect(url: str, flash: str | None = None, kind: str = "success") -> RedirectResponse:
    if flash:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}flash={quote(flash)}&kind={kind}"
    return RedirectResponse(url, status_code=303)


def need_squad(request: Request, ctx: WorkspaceContext) -> HTMLResponse | None:
    if not ctx.has_squad:
        return render("upload.html", request, ctx, reports=[], required_attrs=REQUIRED_ATTRS,
                      notice="Upload a squad export first — every page is built from it.")
    return None


def _records(df: pd.DataFrame | None, cols: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.copy()
    df["_idx"] = df.index if "_idx" not in df.columns else df["_idx"]
    return df.where(pd.notna(df), None).to_dict("records")


def _csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _opt_float(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).replace("£", "").replace(",", "").replace("M", "").replace("K", "").strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    ctx = ctx_for(request)
    if not ctx.has_squad:
        return render("upload.html", request, ctx, reports=[], required_attrs=REQUIRED_ATTRS, notice=None)
    d = ctx.dashboard()
    return render("dashboard.html", request, ctx, d=d, scenarios=store.scenarios(ctx.wid))


# ------------------------------------------------------------------ uploads
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    ctx = ctx_for(request)
    return render("upload.html", request, ctx, reports=[], required_attrs=REQUIRED_ATTRS, notice=None,
                  snapshots=store.list_snapshots(ctx.wid))


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    squad_file: UploadFile | None = None,
    targets_file: UploadFile | None = None,
    registration_file: UploadFile | None = None,
    formation_id: str = Form(""),
    profile_id: str = Form(""),
    transfer_budget: str = Form(""),
    wage_budget: str = Form(""),
):
    ctx = ctx_for(request)
    changes: dict = {}
    if formation_id and formation_id in list_formations():
        changes["formation_id"] = formation_id
        changes["slots"] = None
    if profile_id in PROFILES:
        changes["profile_id"] = profile_id
    tb, wb = _opt_float(transfer_budget), _opt_float(wage_budget)
    if transfer_budget.strip():
        changes["transfer_budget"] = tb
    if wage_budget.strip():
        changes["wage_budget"] = wb
    if changes:
        ctx.save_settings(**changes)

    reports = []
    for kind, up in (("squad", squad_file), ("targets", targets_file), ("registration", registration_file)):
        if up is None or not up.filename:
            continue
        data = await up.read()
        if not data:
            continue
        reports.append(ctx.ingest(kind, data, up.filename))
    if not reports:
        return render("upload.html", request, ctx, reports=[], required_attrs=REQUIRED_ATTRS,
                      notice="No files were selected.", snapshots=store.list_snapshots(ctx.wid))
    errors = [r for r in reports if r.error]
    if errors or not ctx.has_squad:
        return render("upload.html", request, ctx, reports=reports, required_attrs=REQUIRED_ATTRS, notice=None,
                      snapshots=store.list_snapshots(ctx.wid))
    return render("upload.html", request, ctx, reports=reports, required_attrs=REQUIRED_ATTRS, notice=None,
                  snapshots=store.list_snapshots(ctx.wid), go_dashboard=True)


@app.post("/snapshots/delete")
async def delete_snapshot(request: Request, snapshot_id: int = Form(...)):
    store.delete_snapshot(snapshot_id)
    return redirect("/upload", "Snapshot deleted")


@app.post("/snapshots/clear")
async def clear_kind(request: Request, kind: str = Form(...)):
    ctx = ctx_for(request)
    if kind in UPLOAD_KINDS:
        store.clear_kind(ctx.wid, kind)
    return redirect("/upload", f"Cleared {kind} data")


# ----------------------------------------------------------------- settings
@app.post("/settings")
async def update_settings(request: Request):
    form = await request.form()
    ctx = ctx_for(request)
    changes: dict = {}
    if form.get("formation_id") in list_formations():
        changes["formation_id"] = form.get("formation_id")
        changes["slots"] = None
    if form.get("profile_id") in PROFILES:
        changes["profile_id"] = form.get("profile_id")
    if form.get("assumption") in ASSUMPTIONS:
        changes["assumption"] = form.get("assumption")
    for key in ("transfer_budget", "wage_budget"):
        if key in form:
            changes[key] = _opt_float(str(form.get(key)))
    for key in ("seasons", "max_transfers", "squad_size_limit", "max_youth_age", "max_squad", "min_hgn", "u21_age", "max_age", "top_n"):
        if form.get(key, "") != "":
            try:
                changes[key] = int(form.get(key))
            except ValueError:
                pass
    for key in ("min_gain", "min_gain_per_m", "min_margin", "board_sales_percentage"):
        if form.get(key, "") != "":
            v = _opt_float(str(form.get(key)))
            if v is not None:
                changes[key] = v
    if "max_value" in form:
        changes["max_value"] = str(form.get("max_value")).strip()
    if form.get("position_mode") in ("can_play", "cannot_play", "any"):
        changes["position_mode"] = form.get("position_mode")
    if "exclude_unscouted_present" in form:
        changes["exclude_unscouted"] = form.get("exclude_unscouted") in ("on", "1", "true")
    if "locked_players" in form:
        changes["locked_players"] = [n.strip() for n in str(form.get("locked_players")).split("\n") if n.strip()]
    if changes:
        ctx.save_settings(**changes)
    return redirect(str(form.get("next") or "/"))


# --------------------------------------------------------------------- squad
@app.get("/squad", response_class=HTMLResponse)
async def squad_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    a = ctx.analysis
    table = ctx.squad_table()
    role_cols = [ROLES[r].name for r in a.role_ids]
    return render("squad.html", request, ctx, a=a, rows=_records(table), role_cols=role_cols,
                  depth=ctx.depth_chart(), lineup=ctx.pitch_lineup(), weak_idx=a.weak_slots(3))


@app.get("/squad.csv")
async def squad_csv(request: Request):
    ctx = ctx_for(request)
    t = ctx.squad_table()
    if t is None:
        return redirect("/upload")
    return _csv_response(t.drop(columns=["_idx"], errors="ignore"), "squad_scores.csv")


# ------------------------------------------------------------------- compare
COMPARE_COLS = ["Name", "Age", "Club", "Position", "Upgrade Position", "Upgrade Role", "Target Best Score", "Score Low",
                "Score High", "Familiarity", "Upgrade Margin", "Margin Low", "Squad Best Player", "Transfer Value", "Wage",
                "Scouting %", "Needs Scouting", "Personality"]


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    ups = ctx.upgrades() if ctx.has_targets else None
    shortlisted = store.shortlist_names(ctx.wid, "target")
    rows = _records(ups, COMPARE_COLS)
    for row in rows:
        row["shortlisted"] = row["Name"] in shortlisted
    return render("comparison.html", request, ctx, rows=rows, has_targets=ctx.has_targets,
                  total=int(len(ups)) if ups is not None else 0)


@app.get("/compare.csv")
async def compare_csv(request: Request):
    ctx = ctx_for(request)
    ups = ctx.upgrades(top_n=100000) if ctx.has_targets else None
    if ups is None:
        return redirect("/compare")
    return _csv_response(ups[[c for c in COMPARE_COLS if c in ups.columns]], "upgrade_targets.csv")


# --------------------------------------------------------------------- youth
YOUTH_COLS = ["Name", "Age", "Position", "Personality", "Best Role", "Best Role Score", "Score Low", "Score High",
              "Potential Score", "Potential Low", "Potential High", "Growth Signals", "First Team Player",
              "First Team Benchmark", "Gap to First Team", "Est. Seasons to FT", "Readiness", "Transfer Value", "Wage"]


@app.get("/youth", response_class=HTMLResponse)
async def youth_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    df = ctx.youth()
    return render("youth.html", request, ctx, rows=_records(df, YOUTH_COLS))


# ---------------------------------------------------------------------- sell
SELL_COLS = ["Name", "Age", "Position", "Best Slot", "Best Formation Role", "Role Score", "Score Low", "Score High",
             "Rank at Slot", "Potential", "Status", "Recommendation", "Sell Priority", "Reason", "Transfer Value", "Wage",
             "Needs Scouting"]


@app.get("/sell", response_class=HTMLResponse)
async def sell_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    df = ctx.sell()
    rows = _records(df, SELL_COLS)
    counts = df["Recommendation"].value_counts().to_dict() if df is not None and not df.empty else {}
    return render("sell.html", request, ctx, rows=rows, counts=counts, depth=ctx.depth_chart())


# ------------------------------------------------------------------ registration
@app.get("/registration", response_class=HTMLResponse)
async def registration_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    return render("registration.html", request, ctx, reg=ctx.registration_result(), season=ctx.season)


# ------------------------------------------------------------------ strategy
@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    plan = ctx.strategy()
    names = sorted(ctx.analysis.squad_df["Name"].astype(str)) if ctx.analysis else []
    return render("strategy.html", request, ctx, plan=plan, squad_names=names, has_targets=ctx.has_targets)


@app.post("/strategy/save-scenario")
async def strategy_to_scenario(request: Request, name: str = Form("Strategy plan")):
    ctx = ctx_for(request)
    plan = ctx.strategy()
    if not plan:
        return redirect("/strategy", "No plan to save", "warning")
    buys = [a["name"] for a in plan["actions"] if a["type"] == "buy"]
    sells = [a["name"] for a in plan["actions"] if a["type"] == "sell"]
    store.save_scenario(ctx.wid, name, buys, sells, ctx.assumption, int(ctx.settings["seasons"]))
    return redirect("/scenarios", f"Saved '{name}' as a scenario")


# -------------------------------------------------------------------- player
@app.get("/player/{source}/{idx}", response_class=HTMLResponse)
async def player_page(request: Request, source: str, idx: int, partial: int = 0):
    ctx = ctx_for(request)
    if source not in ("squad", "targets") or (r := need_squad(request, ctx)) is not None:
        return r or redirect("/")
    p = ctx.player_detail(source, idx)
    if p is None:
        return redirect("/", "Player not found", "warning")
    tpl = "_player_detail.html" if partial else "player.html"
    return render(tpl, request, ctx, p=p)


# ----------------------------------------------------------------- shortlist
@app.post("/shortlist/add")
async def shortlist_add(request: Request, name: str = Form(...), source: str = Form("target"), note: str = Form(""),
                        next: str = Form("/compare")):
    ctx = ctx_for(request)
    meta = {}
    if source == "target" and ctx.targets is not None:
        i = ctx.find_target_idx(name)
        if i is not None:
            p = ctx.player_detail("targets", i)
            if p and p["best"]:
                meta = {"idx": i, "age": p["age"], "club": p["club"], "position": p["position"], "value": p["transfer_value"],
                        "wage": p["wage"], "best_role": p["best"]["role_name"], "slot": p["best"]["pos"],
                        "score": p["best"]["score"], "lo": p["best"]["lo"], "hi": p["best"]["hi"],
                        "incumbent": p["best"]["incumbent"], "incumbent_score": p["best"]["incumbent_score"],
                        "needs_scouting": p["needs_scouting"]}
    elif source == "squad" and ctx.analysis is not None:
        i = ctx.find_squad_idx(name)
        rec = ctx.analysis.by_idx.get(i) if i is not None else None
        if rec:
            meta = {"idx": i, "age": rec.age, "position": rec.position, "value": rec.transfer_value, "wage": rec.wage,
                    "best_role": rec.best_role_name, "score": rec.best_score, "lo": rec.best_score_lo, "hi": rec.best_score_hi}
    store.shortlist_add(ctx.wid, name, source, note, meta)
    return redirect(next, f"Pinned {name}")


@app.post("/shortlist/remove")
async def shortlist_remove(request: Request, name: str = Form(...), source: str = Form("target"), next: str = Form("/shortlist")):
    ctx = ctx_for(request)
    store.shortlist_remove(ctx.wid, name, source)
    return redirect(next, f"Removed {name}")


@app.get("/shortlist", response_class=HTMLResponse)
async def shortlist_page(request: Request):
    ctx = ctx_for(request)
    items = store.shortlist(ctx.wid)
    return render("shortlist.html", request, ctx, items=items, printable=request.query_params.get("print") == "1")


@app.get("/shortlist.csv")
async def shortlist_csv(request: Request):
    ctx = ctx_for(request)
    items = store.shortlist(ctx.wid)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Name", "Source", "Age", "Club", "Position", "Best role", "Slot", "Score", "Low", "High", "Incumbent",
                "Incumbent score", "Value", "Wage", "Needs scouting", "Note", "Added"])
    for it in items:
        m = it["meta"]
        w.writerow([it["name"], it["source"], m.get("age"), m.get("club"), m.get("position"), m.get("best_role"), m.get("slot"),
                    m.get("score"), m.get("lo"), m.get("hi"), m.get("incumbent"), m.get("incumbent_score"), m.get("value"),
                    m.get("wage"), m.get("needs_scouting"), it["note"], it["added_at"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="shortlist.csv"'})


# ----------------------------------------------------------------- scenarios
def _split_names(raw: str) -> list[str]:
    return [n.strip() for n in raw.replace(",", "\n").split("\n") if n.strip()]


@app.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page(request: Request):
    ctx = ctx_for(request)
    if (r := need_squad(request, ctx)) is not None:
        return r
    scen = store.scenarios(ctx.wid)
    results = []
    for s in scen:
        ev = ctx.scenario(s["buys"], s["sells"], s["assumption"], s["seasons"])
        results.append({"scenario": s, "result": ev})
    shortlist_targets = [i["name"] for i in store.shortlist(ctx.wid) if i["source"] == "target"]
    squad_names = sorted(ctx.analysis.squad_df["Name"].astype(str)) if ctx.analysis else []
    q = request.query_params
    adhoc = None
    if q.get("buys") or q.get("sells"):
        adhoc = ctx.scenario(_split_names(q.get("buys", "")), _split_names(q.get("sells", "")), q.get("assumption"),
                             int(q.get("seasons") or ctx.settings["seasons"]))
    return render("scenarios.html", request, ctx, results=results, shortlist_targets=shortlist_targets,
                  squad_names=squad_names, adhoc=adhoc, adhoc_buys=q.get("buys", ""), adhoc_sells=q.get("sells", ""),
                  adhoc_assumption=q.get("assumption", ctx.assumption))


@app.post("/scenarios")
async def scenario_save(request: Request, name: str = Form(...), buys: str = Form(""), sells: str = Form(""),
                        assumption: str = Form("mid"), seasons: int = Form(3), scenario_id: str = Form("")):
    ctx = ctx_for(request)
    sid = int(scenario_id) if scenario_id.strip().isdigit() else None
    store.save_scenario(ctx.wid, name.strip() or "Scenario", _split_names(buys), _split_names(sells),
                        assumption if assumption in ASSUMPTIONS else "mid", max(1, min(int(seasons), 6)), sid)
    return redirect("/scenarios", f"Saved scenario '{name}'")


@app.post("/scenarios/delete")
async def scenario_delete(request: Request, scenario_id: int = Form(...)):
    ctx = ctx_for(request)
    store.delete_scenario(ctx.wid, scenario_id)
    return redirect("/scenarios", "Scenario deleted")


# ------------------------------------------------------------------- history
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, player: str = ""):
    ctx = ctx_for(request)
    overview = store.history_overview(ctx.wid)
    detail = store.player_history(ctx.wid, player) if player else []
    return render("history.html", request, ctx, overview=overview, detail=detail, player_key=player,
                  snapshots=store.list_snapshots(ctx.wid, "squad"))


# ---------------------------------------------------------------- workspaces
@app.post("/workspace/new")
async def workspace_new(request: Request, name: str = Form("New save")):
    wid = store.create_workspace(name.strip() or "New save")
    store.set_active_workspace(wid)
    return redirect("/upload", f"Created workspace '{name}'")


@app.post("/workspace/switch")
async def workspace_switch(request: Request, workspace_id: int = Form(...), next: str = Form("/")):
    if store.get_workspace(workspace_id):
        store.set_active_workspace(workspace_id)
    return redirect(next)


@app.post("/workspace/rename")
async def workspace_rename(request: Request, name: str = Form(...)):
    ctx = ctx_for(request)
    store.rename_workspace(ctx.wid, name.strip() or "My save")
    return redirect("/upload", "Workspace renamed")


@app.post("/workspace/delete")
async def workspace_delete(request: Request, workspace_id: int = Form(...)):
    store.delete_workspace(workspace_id)
    return redirect("/upload", "Workspace deleted", "warning")


@app.post("/workspace/reset")
async def workspace_reset(request: Request):
    ctx = ctx_for(request)
    for k in UPLOAD_KINDS:
        store.clear_kind(ctx.wid, k)
    return redirect("/upload", "All uploads cleared for this workspace", "warning")


@app.get("/workspace/export")
async def workspace_export(request: Request):
    ctx = ctx_for(request)
    payload = store.export_workspace(ctx.wid)
    name = (ctx.workspace.get("name") or "workspace").replace(" ", "_")
    return Response(json.dumps(payload, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}.fmscorer.json"'})


@app.post("/workspace/import")
async def workspace_import(request: Request, file: UploadFile, name: str = Form("")):
    try:
        payload = json.loads((await file.read()).decode("utf-8"))
        wid = store.import_workspace(payload, name.strip() or None)
    except (ValueError, KeyError, UnicodeDecodeError) as e:
        return redirect("/upload", f"Import failed: {e}", "danger")
    store.set_active_workspace(wid)
    return redirect("/", "Workspace imported")


# ------------------------------------------------------------- configuration
@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    ctx = ctx_for(request)
    forms = list_formations()
    current = ctx.formation
    return render("config.html", request, ctx, formations_all=forms, current=current, layout=pitch_layout(current["slots"]),
                  role_groups=ROLE_GROUPS, slot_positions=SLOT_POSITIONS, custom_roles=load_custom_roles(),
                  custom_profiles=load_custom_profiles(), attrs=sorted(REQUIRED_ATTRS))


@app.post("/config/formation")
async def config_formation(request: Request, formation_id: str = Form(""), name: str = Form(""), slots_json: str = Form("[]"),
                           profile_id: str = Form(""), save_as: str = Form("")):
    ctx = ctx_for(request)
    try:
        slots = json.loads(slots_json)
    except json.JSONDecodeError:
        return redirect("/config", "Invalid formation payload", "danger")
    problems = validate_slots(slots)
    if problems:
        return redirect("/config", "; ".join(problems), "danger")
    slots = [{"pos": str(s["pos"]).upper(), "role": s["role"]} for s in slots]
    pid = profile_id if profile_id in PROFILES else None
    if save_as.strip():
        try:
            f = save_formation(save_as, name or save_as, slots, pid)
        except ValueError as e:
            return redirect("/config", str(e), "danger")
        ctx.save_settings(formation_id=f["id"], slots=None, profile_id=pid)
        return redirect("/config", f"Saved formation '{f['name']}' and made it active")
    ctx.save_settings(slots=slots, formation_id=formation_id or None, profile_id=pid)
    return redirect("/config", "Formation applied to this workspace (unsaved custom)")


@app.post("/config/formation/delete")
async def config_formation_delete(request: Request, formation_id: str = Form(...)):
    ctx = ctx_for(request)
    if not delete_formation(formation_id):
        return redirect("/config", "Built-in formations cannot be deleted", "warning")
    if ctx.settings.get("formation_id") == formation_id:
        ctx.save_settings(formation_id=DEFAULT_SETTINGS["formation_id"], slots=None)
    return redirect("/config", "Formation deleted")


@app.post("/config/role")
async def config_role(request: Request, role_id: str = Form(...), name: str = Form(...), key: str = Form(""),
                      green: str = Form(""), blue: str = Form("")):
    rid = role_id.strip().lower()
    d = {"name": name.strip(), "key": _split_names(key), "green": _split_names(green), "blue": _split_names(blue)}
    problems = validate_role_dict(rid, d)
    if problems:
        return redirect("/config", "; ".join(problems), "danger")
    data = load_custom_roles()
    data[rid] = d
    save_custom_roles(data)
    return redirect("/config", f"Saved role '{name}' ({rid})")


@app.post("/config/role/delete")
async def config_role_delete(request: Request, role_id: str = Form(...)):
    data = load_custom_roles()
    if role_id in data:
        del data[role_id]
        save_custom_roles(data)
        return redirect("/config", f"Removed custom role {role_id}")
    return redirect("/config", "Only custom roles can be deleted", "warning")


@app.post("/config/profile")
async def config_profile(request: Request, profile_id: str = Form(...), name: str = Form(...), description: str = Form(""),
                         changes_json: str = Form("{}")):
    pid = profile_id.strip().lower()
    try:
        changes = json.loads(changes_json or "{}")
    except json.JSONDecodeError as e:
        return redirect("/config", f"Profile changes must be valid JSON: {e}", "danger")
    d = {"name": name.strip(), "description": description.strip(), "changes": changes}
    problems = validate_profile_dict(pid, d)
    if problems:
        return redirect("/config", "; ".join(problems), "danger")
    data = load_custom_profiles()
    data[pid] = d
    save_custom_profiles(data)
    return redirect("/config", f"Saved profile '{name}'")


@app.post("/config/profile/delete")
async def config_profile_delete(request: Request, profile_id: str = Form(...)):
    data = load_custom_profiles()
    if profile_id in data:
        del data[profile_id]
        save_custom_profiles(data)
        return redirect("/config", f"Removed custom profile {profile_id}")
    return redirect("/config", "Only custom profiles can be deleted", "warning")


@app.get("/config/export")
async def config_export(request: Request):
    payload = {"format": "fm-scorer-tactics/1", "roles": load_custom_roles(), "profiles": load_custom_profiles(),
               "formations": {k: v for k, v in list_formations().items() if v.get("custom")}}
    return Response(json.dumps(payload, indent=1, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="fm-scorer-tactics.json"'})


@app.post("/config/import")
async def config_import(request: Request, file: UploadFile):
    try:
        payload = json.loads((await file.read()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return redirect("/config", f"Import failed: {e}", "danger")
    if payload.get("format") != "fm-scorer-tactics/1":
        return redirect("/config", "Not a tactics pack (expected format fm-scorer-tactics/1)", "danger")
    roles = load_custom_roles()
    n_roles = 0
    for rid, d in (payload.get("roles") or {}).items():
        if not validate_role_dict(rid, d):
            roles[rid] = d
            n_roles += 1
    save_custom_roles(roles)
    profs = load_custom_profiles()
    n_prof = 0
    for pid, d in (payload.get("profiles") or {}).items():
        if not validate_profile_dict(pid, d):
            profs[pid] = d
            n_prof += 1
    save_custom_profiles(profs)
    n_form = 0
    for fid, f in (payload.get("formations") or {}).items():
        try:
            save_formation(fid, f.get("name", fid), f.get("slots", []), f.get("profile"))
            n_form += 1
        except ValueError:
            continue
    return redirect("/config", f"Imported {n_roles} roles, {n_prof} profiles, {n_form} formations")


# ----------------------------------------------------------------------- API
@app.get("/api/roles")
async def api_roles():
    return JSONResponse({rid: r.to_dict() for rid, r in ROLES.items()})


@app.get("/api/profiles")
async def api_profiles():
    return JSONResponse({pid: p.to_dict() for pid, p in PROFILES.items()})


@app.get("/api/formations")
async def api_formations():
    return JSONResponse(list_formations())


@app.get("/api/player/{source}/{idx}")
async def api_player(request: Request, source: str, idx: int):
    ctx = ctx_for(request)
    p = ctx.player_detail(source, idx) if source in ("squad", "targets") else None
    if p is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(json.dumps(p, default=str)))


@app.get("/api/scenario")
async def api_scenario(request: Request, buys: str = "", sells: str = "", assumption: str = "", seasons: int = 0):
    ctx = ctx_for(request)
    res = ctx.scenario(_split_names(buys), _split_names(sells), assumption or None, seasons or None)
    if res is None:
        return JSONResponse({"error": "no squad loaded"}, status_code=400)
    return JSONResponse(json.loads(json.dumps(res, default=str)))
