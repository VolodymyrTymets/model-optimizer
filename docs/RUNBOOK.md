# Runbook

> How to run this project, and what goes wrong.

## First run
```bash
# prerequisites: Python 3.9 (the .venv interpreter; no pin file yet), pip, PostgreSQL on localhost:5432
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Apple GPU: .venv-gpu with requirements-gpu.txt
# database: create "model-optimizer-emg" owned by postgres/postgres (default in src/definitions.py)
# data: put the recordings under assets/ (git-ignored, not in the repo)
.venv/bin/python main.py                           # grid experiment
.venv/bin/python train-rare-model.py               # single model
.venv/bin/python export-best-model.py              # restore best step from the DB
```

## Commands
See the command map in `CLAUDE.md` — that is the canonical list. The repo has no tests, linter or build step.

## Undocumented steps everyone knows
- Before a run, set the active signal block in `src/definitions.py` (sound / EMG / EMG-hands / speech): `sr`, `DURATION`, `labels`, `labels_colors` must agree.
- Changing `labels` or `DURATION` points the pipeline at a different `assets/data_set_*` folder; the old one is not deleted.
- Tables are created on `Experiments()` construction (`create_all`); there are no migrations, so a schema change needs a manual DB reset or `ALTER`.

## Common failures
| Symptom | Cause | Fix |
|---------|-------|-----|
| `OperationalError: connection refused` at start | Postgres not running or DB missing | start Postgres, create `model-optimizer-emg` |
| Experiment skipped silently | `experiment.is_finished()` is true for the same settings | change settings or clear that experiment row |
| TensorFlow/Metal import errors | wrong venv (`.venv` vs `.venv-gpu`) | use the venv that matches the requirements file |

## Release
There is no release. Output is an exported SavedModel plus `model.tflite` and `model_settings.json` under `assets/models/`; the `.tflite` and its settings file are what is copied to the device by hand (a human action, rule C4).
