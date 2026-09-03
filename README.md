# FM24 Squad Planner (fm-scorer)

A local web app that turns Football Manager 2024 squad and scouting exports into
transfer-window decisions: a formation-aware Best XI, depth and weak-slot
analysis, upgrade targets ranked by *value for money*, sell/loan/promote
recommendations, youth projections, registration (HGN / U21) planning, and
Plan A / Plan B scenario comparison — all explained in plain English.

Everything runs on your machine; no data leaves it.

## Requirements

- Python 3.10 or newer
- Pinned dependencies in `requirements.txt` (FastAPI 0.115 / Starlette 0.46, pandas 2.3, lxml)

## Install & run

```bash
git clone https://github.com/mcbgun/fm-scorer.git
cd fm-scorer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The first page walks you through the uploads.

Your data (workspaces, snapshots, shortlists, custom roles/formations) is stored
in a SQLite database under `~/.fm-scorer/` (override with `FM_SCORER_DATA_DIR`).
Nothing mutable is written inside the repository.

### Updating to the latest version

After new releases or fixes are pushed, run this from the repository folder:

```bash
./update.sh
```

It safely fast-forwards the current branch, creates `.venv` if necessary, and
installs the pinned dependencies. It refuses to run when local files are
uncommitted, so your work cannot be overwritten. If the app is already running
with `--reload`, it will pick up the updated Python/templates automatically;
otherwise start it again with:

```bash
.venv/bin/uvicorn main:app --reload
```

On Windows, double-click `update.bat` instead. It performs the same update and
dependency installation using the Windows virtual environment.

## Exporting from FM24

The parser expects the attribute columns from Squirrel_plays' FM24 views
(`Acc`, `Pac`, `Sta`, …, `Ref`, `TRO`, `Thr`). Three exports are used:

| Upload | FM screen | Required? | What it powers |
|---|---|---|---|
| **Squad** | *Squad* → view with all attributes, Position, Age, DoB, Transfer Value, Wage, Home-Grown Status (plus Potential / Trn Rat / Injury Susceptibility / Playing Time if you have them) | Yes | Best XI, depth, youth, sell, registration, strategy |
| **Transfer targets** | *Scouting → Players in Range* (or a shortlist) with the same attribute view | Optional | Upgrade targets, strategy purchases, scenarios |
| **Registration** | *Squad → Registration* view (Name + `Inf`) | Optional | Copies the U21/HGN/Wnt/Inj/… status icons onto the squad |

To export any view: select all rows (`Ctrl+A`), then **FM menu → Print Screen →
Web Page** and save the `.html`. Large scouting exports (10k+ players) take a few
seconds to parse.

Attribute *ranges* (`12-16`) from partially-scouted players are preserved: every
score has a low / mid / high band, players with gaps are flagged **Scouting
required**, and you can exclude them from recommendations or plan under
conservative / midpoint / optimistic assumptions.

## What's in the app

- **Dashboard** – Best XI on a pitch, weak slots with top-3 targets, sell and
  youth candidates, registration warnings and budget status.
- **Squad** – depth chart per slot, full sortable table, player detail drawer
  (score breakdown, radar, projection, suitability by slot, history).
- **Targets** – slot-aware upgrade margins (right wingers are not left wingers),
  worst-case margins, value/wage, pin to shortlist.
- **Youth / Sell** – projections using potential stars, training rating, playing
  time, injury susceptibility and age; player-specific reasons for every
  sell / loan / promote / keep recommendation.
- **Registration** – season detected from the export; HGN from *Home-Grown
  Status*, U21 from date of birth (configurable cut-off), other `Inf` icons kept
  as independent statuses.
- **Strategy** – budget-aware plan maximising quality gain per £M with depth
  preserved; low-value moves are listed as skipped, with reasoning.
- **Scenarios / Shortlist** – compare saved plans side by side; CSV and print.
- **History** – every squad upload is a snapshot; track score, value, wage and
  attribute changes per player across seasons.
- **Tactics** – visual formation editor, custom roles and playing-style
  profiles, import/export as a shareable tactics pack. Built-in roles and
  profiles live in `data/*.json`.

## Scoring model

```
score = (Σ key×5 + Σ preferred×3 + Σ useful×1) / (n_key×5 + n_pref×3 + n_useful×1)
slot score = role score × positional familiarity
```

Best XI and benchmarks use a globally optimal (Hungarian) assignment, so a
versatile player never blocks a specialist when a better lineup exists.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Set `FM_SCORER_TEST_SQUAD=/path/to/Squad.html` to run the optional tests against
a real export.

## Known limitations

- Position familiarity comes from the exported *Position* string, which only
  lists Natural/Accomplished positions; FM's finer competence levels are not in
  the export.
- Projections are estimates from age, potential stars and development signals —
  they are shown as bands, not promises.
- Registration rules follow the common 25-man / 8-home-grown pattern; adjust the
  limits on the Registration page for other competitions.
