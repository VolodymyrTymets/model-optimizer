# 1 — Split definitions

Source: "Scope Breakdown" doc, task 1 ("we need split definitions.py depend on task: one for sound, one for emg,
and choose right one via env variable").

## Problem
`src/definitions.py` holds the settings of every signal type in one file. Switching between sound and EMG means
commenting and uncommenting blocks (`sr`, `DURATION`, `labels`, `labels_colors`, and the DB name, `DATA_SET_NAME`,
`VALIDATION_RECORDS_COUNT`). A half-edited file silently points the whole pipeline at the wrong data set
(ARCHITECTURE "Known constraints"). Two git commits (`75921cc`, `b2f2673`) are just this toggle.

## Goal / business value
Choose the task with one environment variable instead of editing source. Supports goal 2 (reproducible
experiments: the active task is a visible input, not an uncommitted edit) without touching accuracy (goal 1).

## Scope
- in: per-task settings for **sound** and **emg** in their own modules; `src/definitions.py` keeps its name and
  its names and picks the module from the env var `MODEL_OPTIMIZER_TASK`; the 27 existing
  `from src.definitions import ...` lines stay unchanged; the variable can also come from a repo-root `.env`
  file, loaded with `python-dotenv` (new runtime dependency, approved by the maintainer on 2026-10-16);
  `.env.example` committed (the real `.env` is already git-ignored and written by the maintainer, never by the
  agent); docs and ADR for the new shape.
- **out (explicit):**
  - new tasks (EMG-hands, speech) as selectable profiles; their commented blocks are recorded in ADR-006, not in code
  - renaming or reshuffling the 24 importing modules, or any change to constants' values
  - schema, DB or `assets/` changes; moving or renaming any data folder
  - ticket 2 and 3 (filtering script and mechanism)
  - any CI change (rule T4)

## Acceptance criteria
- [ ] AC1 Given `MODEL_OPTIMIZER_TASK=sound`, when `src.definitions` is imported, then every public name equals
  its value at `HEAD` (`b2f2673`).
- [ ] AC2 Given `MODEL_OPTIMIZER_TASK=emg`, then every public name equals the active EMG values of commit
  `75921cc` (`sr=8820`, `DURATION=0.05`, `labels=['noise','stimulation']`, DB `model-optimizer-emg`,
  `DATA_SET_NAME='data_set_emg'`, `VALIDATION_RECORDS_COUNT=10`).
- [ ] AC3 Given the variable is unset, then the sound profile loads (today's behaviour).
- [ ] AC4 Given an unknown value (e.g. `foo`), then import raises `ValueError` naming the variable, the bad
  value and the valid values; nothing falls back silently.
- [ ] AC5 Given either value, every module that imports `src.definitions` still imports (no edit to those modules).
- [ ] AC6 Shared constants (`ASSETS_PATH`, `VERBOSE`, `EMULATE_MODE`, `SKIP_FILTER`, `frame_length`, `hop_length`,
  `n_mels`, `n_mfcc`, `sub_sets`, `keet_prefix`) are defined once, not duplicated per task.
- [ ] AC8 Given a repo-root `.env` containing `MODEL_OPTIMIZER_TASK=emg` and the variable not set in the shell,
  then the EMG profile loads from any working directory; given both, the shell value wins; given no `.env`, no
  error and AC3 applies.
- [ ] AC9 `python-dotenv` is pinned at `1.2.1` in `requirements.txt` and `requirements-gpu.txt` and imports in
  `.venv` (Python 3.9.6; 1.2.2+ dropped 3.9).
- [ ] AC7 ARCHITECTURE, RUNBOOK, GLOSSARY, a new ADR and rule P2 describe the new shape (rule E1).

## Edge cases
| Case | Expected behaviour | Decided by |
|------|--------------------|-----------|
| Variable set to empty string | treated as invalid → AC4 error (not as unset) | assumption |
| `FRAGMENT_LENGTH` | derived in `definitions.py` from the active `sr`/`DURATION`, never stored per task | AC1/AC2 |
| Logger importing `definitions` (`VERBOSE`) | `definitions.py` must not import `Logger` (circular) | code |
| Existing cooked data sets / models | same labels + duration → same fingerprint → still found | ARCHITECTURE |

## Open questions
| # | Question | Assumed answer | Needs a human? |
|---|----------|----------------|----------------|
| 1 | Env var name and values | `MODEL_OPTIMIZER_TASK` = `sound` \| `emg` | yes, quick |
| 2 | Default when unset | `sound` (matches `HEAD`, no behaviour change). Alternative: fail when unset, safer against a silent wrong task | yes |
| 3 | EMG values source | the active EMG block of `75921cc`; the ticket does not list them | yes, confirm |
| 4 | Where do the EMG-hand and speech commented blocks go | not copied into code (rule D6); their label lists and colours are recorded in ADR-006 and stay in git history (`b2f2673`), so nothing is lost | yes, confirm |
| 6 | `.env` support | answered 2026-10-16: add `.env.example` + `python-dotenv` (dependency approved, rule C2) | no |
| 5 | Rule P2 says tunables live in `src/definitions.py` only; `CLAUDE.md` is an ask-first path | reword P2: `definitions.py` stays the only import surface, per-task values live in `definitions_<task>.py`; maintainer approves the wording | **yes (F2)** |
