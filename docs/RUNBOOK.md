# Runbook

> How to run this project, and what goes wrong.

## First run
```bash
# prerequisites: Python 3.9 (the .venv interpreter; no pin file yet), pip, PostgreSQL on localhost:5432
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Apple GPU: .venv-gpu with requirements-gpu.txt
# database: create one per task you run, owned by postgres/postgres: "model-optimizer-sound" and "model-optimizer-emg" (src/definitions_<task>.py)
cp .env.example .env                               # then set MODEL_OPTIMIZER_TASK=sound or emg (unset = sound)
# data: put the recordings under assets/ (git-ignored, not in the repo)
.venv/bin/python main.py                           # grid experiment
.venv/bin/python train-rare-model.py               # single model
.venv/bin/python export-best-model.py              # restore best step from the DB
.venv/bin/python filter-data-set.py <model_name>   # filter the cooked data set with an exported model; add --dry-run first
```

## Commands
See the command map in `CLAUDE.md` — that is the canonical list. The repo has no tests, linter or build step.

## Undocumented steps everyone knows
- Before a run, choose the task: `MODEL_OPTIMIZER_TASK=emg .venv/bin/python main.py`, or put the line in the repo-root `.env`. A value set in the shell wins over `.env`. The values per task are in `src/definitions_<task>.py`; EMG-hands and speech are not selectable (ADR-006 keeps their labels).
- Changing `labels` or `DURATION` points the pipeline at a different `assets/data_set_*` folder; the old one is not deleted.
- Tables are created on `Experiments()` construction (`create_all`); there are no migrations, so a schema change needs a manual DB reset or `ALTER`.

- Filtering (`filter-data-set.py`): run it with `--dry-run` first, it logs every file it would move (name, from and to label) plus counts, and moves nothing. That output is recorded data: keep it out of prompts, issues and PRs. The real run moves files in place with no undo, on `train` and `test` (so validation records inherit the model's labels), and writes `__filtered__`; a second run is skipped while that marker exists. The cooked data set must exist (run `main.py` once) and the next `main.py` run re-splits it, discarding the filtering.

## Common failures
| Symptom | Cause | Fix |
|---------|-------|-----|
| `OperationalError: connection refused` at start | Postgres not running or DB missing | start Postgres, create the database of the active task (`model-optimizer-sound` or `model-optimizer-emg`, see `src/definitions_<task>.py`) |
| Experiment skipped silently | `experiment.is_finished()` is true for the same settings | change settings or clear that experiment row |
| `ValueError: MODEL_OPTIMIZER_TASK='…' is not valid` at import | typo in the shell or `.env` value | use `sound` or `emg` |
| Wrong data set or database used | variable not set, so the default `sound` applied | set `MODEL_OPTIMIZER_TASK` in the shell or `.env` |
| TensorFlow/Metal import errors | wrong venv (`.venv` vs `.venv-gpu`) | use the venv that matches the requirements file |

## Release
There is no release. Output is an exported SavedModel plus `model.tflite` and `model_settings.json` under `assets/models/`; the `.tflite` and its settings file are what is copied to the device by hand (a human action, rule C4).
