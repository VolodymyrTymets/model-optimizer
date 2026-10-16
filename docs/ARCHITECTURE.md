# Architecture

> What the system **is** and where code belongs. Structure truths only — no style preferences (those live in `.claude/rules/`).
> Update in the same PR as any structural change (rule E1).

## Repo layout
| Path | What lives here | Deployable? | Owner |
|------|-----------------|-------------|-------|
| `main.py` | Entry: grid experiment run (`Experiments.run`) over audio-feature types x augmentations x layer/optimizer choices | no (local script) | maintainer |
| `train-rare-model.py` | Entry: single hand-built Conv2D model for a rare-class data set; trains, exports plots, validates, labels records | no | maintainer |
| `export-best-model.py` | Entry: `ModelRestorer.restore_best_step()` — restores the best step from the DB | no | maintainer |
| `filter-data-set.py` | Entry: `DataSetFilter` with `LocalModelLoader` moves mislabeled fragments of the cooked data set to the label a chosen exported model predicts (`--dry-run` to preview) | no | maintainer |
| `src/definitions.py` | The one import surface for tunables: shared constants (frame/hop, `ASSETS_PATH`, flags), `FRAGMENT_LENGTH`, and the active task's values; picks the task from `MODEL_OPTIMIZER_TASK` (ADR-006) | no | maintainer |
| `src/definitions_sound.py`, `src/definitions_emg.py` | Per-task values: `sr`, `DURATION`, `labels`, `labels_colors`, `DATA_SET_NAME`, `DEFAULT_DATABASE_URL`, `VALIDATION_RECORDS_COUNT` | no | maintainer |
| `.env.example` | Documents `MODEL_OPTIMIZER_TASK`; the real `.env` is git-ignored | no | maintainer |
| `src/experiments`, `src/experiment` | Orchestration: one `Experiment` = a model-search run with steps; `Experiments` loops feature x augmentation combinations | no | maintainer |
| `src/data_set` | Cook (`DataSetCooker`), augment, split, filter, import (`DataSetImporter`) of the audio/EMG data set | no | maintainer |
| `src/utils/audio_features` | Feature strategies (`wave, ae, rms, zcr, fft, stft, ber, sc, bw, mel, mfcc`) behind `IAFStrategy` + factory | no | maintainer |
| `src/model_builder`, `model_tuner`, `model_trainer`, `model_schema` | Build, tune (layer/units search), train Keras models; schema enums (`LayerType`, `OptimizerType`, …) | no | maintainer |
| `src/model_validator` | Validate on held-out records; per-record accuracy, result parsing, labeling | no | maintainer |
| `src/model_exporter`, `src/model_restorer` | Save TF SavedModel, float32 TFLite + settings sidecar, plots; restore best step | no | maintainer |
| `src/database` | SQLAlchemy `Base`/tables and `DBClient` singleton | no | maintainer |
| `src/assets_service` | Resolves every path under `assets/` (data set, experiment, models, validation records) | no | maintainer |
| `assets/` | **Git-ignored**, 1.6 GB: raw/cooked data sets, per-experiment output, exported models | no | local only |
| `requirements.txt`, `requirements-gpu.txt` | Pinned deps; the GPU file adds `tensorflow-metal` (Apple) | no | maintainer |

## Where data truth lives
- **Experiment history** (experiments, details, data-set details, steps, model schemas, layers): Postgres, via `src/database/schema.py`. Authoritative for "which run is best".
- **Data sets and model artefacts**: files under `assets/`, named by a fingerprint of labels + duration (+ feature type for models). Derived: can be rebuilt from the raw recordings by `DataSetCooker`.
- **Signal shape and labels**: `src/definitions_<task>.py`, read through `src/definitions.py` — changing `sr`/`DURATION`/`labels` invalidates cooked data sets and exported models.

## Boundaries and contracts
| Boundary | Format | Source of truth | Generated output (never hand-edited) | Regeneration command |
|----------|--------|-----------------|--------------------------------------|----------------------|
| Exported model | TF SavedModel with a `get_settings` signature, plus `model.tflite` and `model_settings.json` beside it (ADR-005) | `ModelExporter` / `ModelInstance` | `assets/models/**`, `assets/experiment-*/model_*` | `.venv/bin/python main.py` / `train-rare-model.py` |
| Database tables | SQLAlchemy models, `create_all` (no migrations) | `src/database/schema.py` | Postgres tables | created on `Experiments()` construction |

The exported model is consumed by an external device/app that is **not in this repo** — see "Known constraints".

## Environments
| Env | URL | Database | Who may touch it |
|-----|-----|----------|------------------|
| local | n/a | one per task: `postgresql://postgres:postgres@localhost:5432/model-optimizer-sound` (task `sound`) and `.../model-optimizer-emg` (task `emg`), set in `definitions_<task>.py` | anyone |
| staging | none | none | — |
| production | none | none | **not the agent** (rule C5) |

## Key flows
### 1. Grid experiment (`main.py`)
`Experiments()` creates tables → for each `AFTypes` x `ArgumentationTypes`: build `ExperimentDataSetDetails` + feature strategy → skip if the experiment is already finished → `DataSetCooker.prepare` (no `__filtered__` marker: delete and re-split `assets/data_set_*`, then filter once with `assets/models/model-<labels>-<duration>-<af_type>` if that folder exists, which only `export-best-model.py` writes; marker present: keep the data set; then, if marked, remove old augmented files and augment again, and regenerate validation records if they are older than the marker) → `DataSetImporter` → `Experiment.start()` runs tuner steps, trains, records results in Postgres → `finish()`. Filtering happens once per data-set folder inside `prepare()`, never per step (ADR-008); a first run, with no exported model, stays unfiltered and is not augmented. `filter-data-set.py` (flow 4) does the same by hand.

### 2. Single rare model (`train-rare-model.py`)
Create/lookup experiment → cook data set → import train/val/test → build fixed Conv2D model → fit with early stopping → export model plot + training plot → `ModeValidator.validate` on `assets/validation-records` → `ModelRecordLabeler` writes labelable records.

### 3. Export best (`export-best-model.py`)
`ModelRestorer.restore_best_step()` picks the best step from the DB and restores it.

### 4. Filter by model (`filter-data-set.py <model_name>`)
`<model_name>` is a folder under `assets/models/` (`model-<labels>-<duration>-<af_type>`). `AssetsService.parse_model_name` / `get_data_set_path_for_model` map it to `assets/<DATA_SET_NAME>_<labels>-<duration>` (no DB) → the script checks labels and duration against `src/definitions.py` → `LocalModelLoader` loads the SavedModel → `DataSetFilter` plans every move, refuses if any would overwrite a file, then moves fragments in place on `train` and `test` and writes `__filtered__`. files with `keep` in the name and augmented files are skipped. The next `main.py` run keeps this data set (flow 1). The log lists each move (`set/label/file -> set/new label`) and counts; keep that output out of prompts, issues and PRs (rule G1).

## External integrations
| Service | Purpose | Failure mode | Sandbox available? |
|---------|---------|--------------|--------------------|
| PostgreSQL (local) | experiment tracking | `Experiments()` fails at construction; nothing runs | yes (local) |

## Known constraints and landmines
- `DEFAULT_DATABASE_URL` embeds local `postgres:postgres` credentials in source; fine for a local DB, must never point at a shared host (rules D8, C5).
- `MODEL_OPTIMIZER_TASK` unset means `sound`: forgetting it on an EMG run silently uses the sound database and `data_set_sound`. An unknown value raises `ValueError` at import.
- `assets/` fingerprints do not include the model settings; an old cooked data set is reused if labels + duration match.
- `train-rare-model.py:63` uses `input_shape`, which is not defined anywhere in the file: `build_model` raises `NameError` as written.
- Filtering by model moves files in place and cannot be undone. `main.py` does it automatically in `prepare()` when `assets/models/model-<labels>-<duration>-<af_type>` exists (`train-rare-model.py` opts out); the only preview is `filter-data-set.py --dry-run` beforehand. `DataSetCooker.prepare()` keeps a data set that carries `__filtered__`, so a filtered data set lives until the maintainer deletes its folder (changed raw recordings are not picked up); an unfiltered one is deleted and re-split at the start of every experiment. `test` is filtered too, so validation records, built from `test`, carry the filter model's labels: this can inflate held-out accuracy.
- Augmentation is live again on a filtered data set and lives in the shared `data_set_<fingerprint>` folder: `export-best-model.py` (`ModelRestorer`) retrains the best step on whatever augmentation the last `prepare()` left there, and `train-rare-model.py` (`[nothing]`) strips it. Augmented copies sit next to their base fragment, so the importer's random train/validation split can put a copy in validation and its base fragment in train (validation accuracy looks better than it is; record accuracy is unaffected). Neither is fixed here (rule B4).
- `assets/validation-records` is shared by every data set and task, while the `__filtered__` marker belongs to one data set: `_records_outdated()` is only right when one data set owns the records. Switching task (or data set) after filtering another one can leave records built from the other task's `test` set.
- Python 3.9 + `numpy 2.0.2` + `tensorflow 2.20` + `keras 3.10` are pinned together; bumping any one is its own PR.
- Typo'd module names are real import paths (`mode_trainer`, `mode_builder`, `mode_validator_interfcace`, `expirement_summarize_service`); do not "fix" them in a feature PR (rule C1).
