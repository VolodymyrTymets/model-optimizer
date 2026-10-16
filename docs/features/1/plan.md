# 1 — Split definitions: implementation plan

<!-- A human writes this line to approve. Until it is here, guard-plan-approval blocks writes to code. -->
Approved-by: volodymyr · 2026-10-16

Pattern followed: no existing env-driven config in `src/` (grep for `environ`/`getenv` finds nothing), so the
shape is new but small. It follows rule P2 and D2: `src/definitions.py` stays the single module everything
imports from (24 importing modules untouched), flat `src/definitions_<task>.py` siblings like the other flat modules in
`src/`. Python-core rule 8 (config comes from the environment or the repo's config module) is the basis.
Because this changes P2's "one file" wording, an ADR is required (rule D1).

## Contract changes
- Import surface: `src.definitions` exposes the same names with the same types. Additive: `TASK` (the active
  profile name). Nothing removed or renamed.
- New runtime input: env var `MODEL_OPTIMIZER_TASK` (`sound` | `emg`, default `sound`), from the shell or a
  repo-root `.env`; the shell wins (`load_dotenv(override=False)`, verified in the python-dotenv docs).
  Running EMG now needs `MODEL_OPTIMIZER_TASK=emg .venv/bin/python main.py` or `.env` with that line.
- New runtime dependency: `python-dotenv==1.2.1` (last release supporting Python 3.9), in `requirements.txt` and
  `requirements-gpu.txt`. Approved by the maintainer on 2026-10-16 (rule C2); both files are ask-first paths.
- Persisted formats, DB schema and `assets/` layout: unchanged (fingerprints depend on labels + duration only).

## Layout after the change
- `src/definitions.py`: shared constants; reads the env var; picks the profile from a fixed dict (no dynamic
  import of arbitrary names); explicit re-assignment of the per-task names; derives `FRAGMENT_LENGTH`.
  Imports nothing from `src.utils` (Logger imports `definitions` for `VERBOSE`).
- `src/definitions_sound.py`: `DEFAULT_DATABASE_URL` (`...model-optimizer-sound`), `DATA_SET_NAME`
  (`data_set_sound`), `VALIDATION_RECORDS_COUNT` (40), `sr` (44100), `DURATION` (0.5), `labels`, `labels_colors`.
- `src/definitions_emg.py`: same names with the EMG values of `75921cc`.
- EMG-hands and speech blocks (commented out today): not copied into code, because rule D6 forbids commented-out
  code. Their labels and colours go into ADR-006 as reference data, and remain in git history (`b2f2673`).
- `definitions.py` imports with private aliases (`import os as _os`, `from pathlib import Path as _Path`,
  `from dotenv import load_dotenv as _load_dotenv`, `typing` names as `_`-names) so the only new public name is
  `TASK`.

## Requirements (ordered, each independently verifiable)

### R0 — Dependency and `.env.example` (S)
- files: `requirements.txt` (change: add `python-dotenv==1.2.1` in alphabetical position), `requirements-gpu.txt`
  (same), `.env.example` (add: `MODEL_OPTIMIZER_TASK=sound` and a comment listing `sound` | `emg`)
- layer: dependencies / config
- not touched: `.env` (protected, the maintainer creates it by copying the example); `.gitignore` already lists it
- executed how: `.venv/bin/pip install -r requirements.txt` (pip only, rule T3), then
  `.venv/bin/python -c "import dotenv; print(dotenv.__file__)"` and `.venv/bin/pip show python-dotenv`
- evidence: pip output and the `Version: 1.2.1` line (AC9). `.venv-gpu` is not installed here; the GPU file gets
  the same pin and is listed under "Not verified".

### R1 — Per-task profiles and the selector (M)
- files: `src/definitions_sound.py` (add), `src/definitions_emg.py` (add), `src/definitions.py` (change)
- layer: config
- type hints: module constants get annotations where the new code defines them (`labels: List[str]`,
  `labels_colors: Dict[str, str]`); on 3.9 use `typing` forms (python-core rule 1)
- `.env` loading: `load_dotenv(Path(__file__).resolve().parent.parent / '.env')` at the top of `definitions.py`,
  explicit path so the launch directory does not matter (python-core rule 7), `override=False` so a shell
  variable wins; a missing file is not an error. The one import-time side effect is the config read this module
  already is.
- error: unknown or empty value raises `ValueError("MODEL_OPTIMIZER_TASK='foo' is not valid; use one of: emg, sound")`
  (rule D5); no fallback
- test: none exists in `src/` (rule P6); verified by the comparison script below
- executed how (scratchpad script, never touches `assets/` or Postgres):
  1. snapshot A = `git show HEAD:src/definitions.py` executed into a dict; snapshot B = `git show
     75921cc:src/definitions.py` likewise
  2. for each of `sound`, `emg`: import `src.definitions` in a fresh `.venv/bin/python -I` subprocess with the env
     var set, dump all public names, diff against A (sound) / B (emg) → AC1, AC2, AC6
  3. unset var → equals A (AC3); `foo` and empty string → `ValueError` text printed (AC4)
  4. AC8: the agent cannot create a repo-root `.env` (protected). `definitions.py` does `from dotenv import
     load_dotenv`; the subprocess wrapper imports `dotenv` first and replaces `dotenv.load_dotenv` with a
     small wrapper that records the path it receives and loads the scratchpad `.env` instead (a bound `partial`
     would raise on the positional path), then imports `src.definitions`. Run from `cwd` = the scratchpad and from
     the repo root; the recorded path must equal `<repo>/.env` in both. Checks: file only → emg; file + shell value → shell wins; no file → sound
  Not verified: the real repo-root `.env` path; the maintainer creates it (`cp .env.example .env`) and runs once.
- evidence: the diff output for each case, with the excluded-name list printed (`TASK` only; dunder names), and
  the error text

### R2 — Import smoke over every consumer (S)
- files: none changed
- executed how: scratchpad script builds the module list at run time from `grep -rln "from src.definitions" src`
  (24 modules today, 27 import lines) and imports each in a fresh process, once per task value.
  `train-rare-model.py` is left out on purpose (hyphenated name, import-time side effects, known `NameError`);
  `main.py` is not run here. Then, per task, in a fresh process: print `AssetsService(experiment_id=0)
  .out_data_set_name` (constructor makes no query) and `DBClient()._database_url` (`create_engine` is lazy, no
  connection). Expect `data_set_sound` / `...model-optimizer-sound` and `data_set_emg` / `...model-optimizer-emg`.
  No Postgres, no listing or reading of `assets/` (P5).
  The `train-rare-model.py` consumer is covered by an `ast` check: every name it imports from `src.definitions`
  exists on the module for both tasks. The DB URL is printed with the password masked (rule D8).
- evidence: pass/fail table, 24 modules x 2 tasks (AC5), the `ast` check, and the two printed pairs
- not verified: the full fingerprinted data-set path (`get_data_set_path` needs Postgres and an experiment row)
- not verified here: a real `main.py` run needs local Postgres and the recorded data under `assets/` (P5, P6).
  Reported under "Not verified"; the maintainer runs `MODEL_OPTIMIZER_TASK=emg .venv/bin/python main.py` once.

### R3 — Docs, ADR and rule wording (S)
- files: `docs/decisions/ADR-006-task-profiles-via-env.md` (add, from `ADR-000-template.md`),
  `docs/ARCHITECTURE.md` (change: layout row for `src/definitions*.py`, drop the "mixes active and commented-out
  configs" landmine, add the "default is sound when unset" landmine; fix the stale Environments row: one local
  database URL per task), `docs/RUNBOOK.md` (change: replace the
  "set the active signal block" step with the env var, add a failure row for the `ValueError`; fix the stale
  "First run" line: one database per task, `model-optimizer-sound` and `model-optimizer-emg`),
  `docs/DOMAIN_GLOSSARY.md` (add "Task" / "profile"), `Readme.md` if it describes the switch (check first),
  `CLAUDE.md` rule P2 (ask-first path: applies the wording the maintainer approved above)
- executed how: re-read each doc against the final code; run the RUNBOOK command line from a clean shell
  (R1 script) so the documented command is the one that was executed
- evidence: `git diff --stat` of docs

## Risks and unknowns
1. **Silent wrong task.** Unset var → sound, so an EMG user who forgets it trains on sound settings and writes into
   the sound DB and `data_set_sound`. Cheapest removal: the open question 2 — fail when unset — or print `TASK`
   at the top of the entry scripts (not in this PR without approval, rule C1).
2. **Rule P2 / constitution.** The split changes a maintainer-confirmed rule in an ask-first file. The wording is
   approved by the maintainer before implementation (see "Decisions" above, F2), not after.
3. **EMG values are recovered from git history** (`75921cc`), not from the ticket. If a newer EMG setup was
   used locally, AC2 compares against stale values. Cheapest removal: the maintainer confirms the list in
   open question 3 before R1. Existing `assets/data_set_emg` was not inspected (P5).

## Docs to update in the same PR (rule E1)
ARCHITECTURE, RUNBOOK, GLOSSARY, ADR-006, `CLAUDE.md` P2 (approval needed), `Readme.md` if relevant; RUNBOOK documents `cp .env.example .env`.

## Estimate
R0 S, R1 M, R2 S, R3 S. One **draft** PR, one requirement (rule A4), branch `feat/1-split-definitions`.

## Decisions to record before /implement (gate blocker)
Open questions 1, 2, 3, 4 and 5 of the spec need the maintainer's answers. They are written here, next to the
`Approved-by:` line, when the plan is approved:
- `.env` + `python-dotenv==1.2.1` approved by the maintainer, 2026-10-16 (in session)
- env var `MODEL_OPTIMIZER_TASK` = `sound` | `emg`: approved 2026-10-16
- default when unset = `sound`: approved 2026-10-16
- EMG values from `75921cc` (sr 8820, DURATION 0.05, noise/stimulation, DB `model-optimizer-emg`,
  `data_set_emg`, 10 validation records): approved 2026-10-16
- EMG-hands/speech blocks recorded in ADR-006 only: approved 2026-10-16
- P2 wording, approved 2026-10-16, applied by R3 as written: "Tunables (`sr`, `DURATION`, `labels`,
  `ASSETS_PATH`, frame sizes) are imported only from `src/definitions.py`. Task-specific values live in
  `src/definitions_<task>.py` (`sound`, `emg`), chosen by `MODEL_OPTIMIZER_TASK` (shell or repo-root `.env`).
  Do not scatter constants or add another selector."

## Definition of done
- [ ] AC1–AC9 each shown by captured output or a doc diff
- [ ] No consumer module edited; values unchanged for both tasks
- [ ] "Not verified": real pipeline run (Postgres + `assets/`), GPU
