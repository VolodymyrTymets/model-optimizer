# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A biomedical/speech audio-ML pipeline that automatically searches Keras model architectures
(layer types, units, activations, regularizers, optimizer, loss) for a WAV-clip classifier, and
tracks every experiment/step/schema/weights/plot it tries in Postgres so a run is resumable and
never repeats work. Two subsystems:

1. **Experiment/auto-tuning pipeline** (`main.py`, `src/experiments`, `src/experiment`,
   `src/model_tuner`, `src/model_builder`, `src/model_trainer`, `src/model_validator`,
   `src/model_exporter`, `src/model_restorer`, `src/model_schema`, `src/database`,
   `src/assets_service`) — the current focus, and fully implemented (not stubs).
2. **Dataset preparation + audio features** (`src/data_set`, `src/utils/audio_features`) — turns
   raw CSV/WAV recordings into fragmented, augmented, feature-image datasets. `src/definitions.py`
   currently has `labels` set to a 9-class speech-command set (`noise, down, up, go, left, right,
   stop, yes, no`); the original RLN 3-class set (`noise, stimulation, breath`) and an EMG gesture
   set are kept commented out next to it as swap-in alternatives.

## Setup

No lint/test/build tooling is configured. Two virtualenvs exist (Python 3.9): `.venv` (CPU,
`requirements.txt`, TF 2.20) and `.venv-gpu` (Apple Silicon, `requirements-gpu.txt`, TF 2.17.1 +
`tensorflow-metal`). Install from whichever `requirements*.txt` matches the venv you activate.

The pipeline requires a **running local Postgres** reachable at the URL in
`src/definitions.py:DEFAULT_DATABASE_URL` (`postgresql://postgres:postgres@localhost:5432/model-optimizer-speech`).
`DBClient.create_database()` only creates tables (`Base.metadata.create_all`) — the database itself
must already exist (e.g. `createdb model-optimizer-speech`) before running anything.

```bash
source .venv/bin/activate          # or .venv-gpu/bin/activate on Apple Silicon
pip install -r requirements.txt    # or requirements-gpu.txt
python main.py                     # runs the full auto-tuning experiment via Experiments.run(...)
python train-rare-model.py         # standalone one-off train/validate/label run, bypasses the DB/tuner pipeline
python export-best-model.py        # ModelRestorer: re-cooks the best DB-tracked step and exports it as a SavedModel
```

## Architecture

### Config hub — `src/definitions.py`
Central constants imported across both subsystems: `sr`, `DURATION`/`FRAGMENT_LENGTH`,
`frame_length`/`hop_length`, `n_mels`/`n_mfcc`, `labels` + `labels_colors`, `sub_sets =
['train','test']`, `DEFAULT_DATABASE_URL`. `DATA_SET_TYPE` (`'audio'` | `'image'`) switches the whole pipeline: in `'image'` mode the dataset is
read as-is from `assets/<DATA_SET_NAME>/{train,test}/<label>/*.jpg` (`ImageDataSetImporter`, `IMAGE_SIZE`),
`af_strategy` is `None`, there is no split/filter/augment/validation-recording step and **no record
accuracy** (`record_accuracy` stays 0 and `accuracy_delta` = validation accuracy); the DB stores the
placeholders `af_type='none'`, `duration=0`. `main.py` picks its search space by this constant.
Also two pipeline toggles worth knowing about:
`EMULATE_MODE` (short-circuits model build/train/validate with mocks — for fast end-to-end
pipeline smoke tests without real training) and `SKIP_FILTER` (skips the self-filtering pass
described below).

### Auto-tuning search — `src/model_tuner/`
`ModeTuner` runs three stages in order, each returning a refined `ModelSchema`:
`rare_tuning()` (one layer, low units — the coarse first pass; this is where "rare" in filenames
and the `rare-model` branch name comes from, unrelated to "RLN") → `layers_tuning()` (adds each
configured `LayerType` one at a time, first individually then stacked together, searching
activation/regularizer/units per layer) → `final_tuning()` (re-sweeps optimizer/loss on the
settled architecture). Unit-count search (`LayerTuner._get_best_units`) is a recursive binary
search over a power-of-two range derived from `ExperimentDetails.units_range`. Every candidate
schema is submitted as one `ExperimentStep.run(schema, epochs)`.

### Experiment/DB tracking — `src/database/`, `src/experiment/`
`DBClient` is a process-wide singleton SQLAlchemy engine/session factory; `session_scope()` is the
standard commit/rollback/close context manager. `schema.py` declares `ExperimentModel` →
`ExperimentDetailsModel`/`ExperimentDataSetDetailsModel` (the search-space config, serialized as
comma-joined strings) → `ExperimentStepModel` → `ModelSchemaModel` → `ModelLayerModel`, plus
`WeightsModel` (pickled weights stored **in the DB**, not on disk) and `ImageModel`/
`RecordResultModel` for plots and per-record validation results.

This makes reruns idempotent at two levels: `ExperimentModelService.get_current_experiment`
looks up an existing experiment by exact-match on serialized details before creating a new row,
and `ExperimentStep.run` looks up an existing step by schema fingerprint before training — a step
already finished with `accuracy_delta > 0` is skipped. `Experiment.start()` runs the three tuning
stages then `_finish_unfinished_steps()` to pick up anything interrupted mid-run.

`Experiments` (plural, `src/experiments/experiments.py`) is the top-level orchestrator `main.py`
calls: for each `(af_type, argumentation_type)` combination it creates/looks-up an `Experiment`,
prepares the dataset, and (if `train=True`) runs it.

### Asset paths — `src/assets_service/`
`AssetsService` computes fingerprinted, reusable paths from an experiment's DB-stored details:
the dataset dir is fingerprinted on sorted labels + duration (`get_data_set_path`), the model dir
additionally on `af_type` (`get_models_path`/`get_model_fingerprint`) — so two experiments with
matching data-relevant settings share prepared data on disk even if other hyperparameters differ.

### Dataset preparation — `src/data_set/`
All stages subclass `DataSetFileWorker` (file I/O + `read_data_set()` generator) in
`src/data_set/utils/`. On-disk tree: `assets/<dataset>/<set_name>/<label>/*.wav`.
`DataSetCooker.prepare()` runs: split (`data_set_splitter`, fixed-duration chunks) → filter
(`data_set_filter`, self-training cleanup, see below) → augment (`data_set_transformer`:
normalize/pitch/time-shift/time-stretch via `ArgumentationTypes`) → generate synthetic validation
recordings + `*.annotation.json` timestamps (`data_set_record_generator`). `except_sets`/
`except_labels` restrict an op to a subset (e.g. augment train but not test; never touch `noise`).

**Self-filtering** (`data_set_filter.py`): once a model exists for the current experiment
settings — from the DB via `InMemoryModelLoader`, or from a saved model on disk via
`LocalModelLoader` — `DataSetFilter.filter()` re-predicts every raw (non-augmented) training file
and relocates any whose predicted label disagrees with its folder, i.e. the model prunes/relabels
its own training data. Gated by `SKIP_FILTER` and a `__filtered__/accuracy.txt` marker so it only
reruns once the best step's `accuracy_delta` has improved.

### Model build/train/validate/export
- `src/model_builder/mode_builder.py` — `ModeBuilder.build_model(schema, train_ds)` assembles a
  `tf.keras.Sequential`, mapping each `LayerSchema` to Keras layers (`Conv` gets a
  reshape→Conv2D→MaxPool2D→reshape block; `GRU`/`Dense` map directly) and enums to TF
  functions/classes.
- `src/model_trainer/mode_trainer.py` — `ModeTrainer.train(...)`, `EarlyStopping` on `val_loss`.
- `src/model_validator/` — `ModeValidator.validate(...)` returns **two** accuracy numbers: keras
  `.evaluate()` accuracy, and a stricter "record accuracy" from `ModelRecordEvaluator`, which
  slides `FRAGMENT_LENGTH` windows over each synthetic test recording and checks the model's
  per-window prediction against the annotated label timestamp ranges.
- `src/experiment/experiment_step/experiment_step_model_cooker.py` — `ExperimentStepModelCooker`
  is the actual per-step unit of work: build → restore-or-export weights
  (`ModelWeightsExporter`, pickled into `WeightsModel`) → train → (validated by the caller).
- `src/model_exporter/` — `ModelExporter.export_model()` wraps model+labels+af-type+duration in a
  `ModelInstance` and saves a `tf.saved_model` (plus `export_model_plot`/`export_training_plot`
  matplotlib PNGs). `src/model_restorer/model_restorer.py` — `ModelRestorer.restore_best_step()`
  looks up the best-accuracy `ExperimentStepModel` in the DB, re-cooks it (reusing cached
  weights), and exports it — this is `export-best-model.py`.

### Audio features — `src/utils/audio_features/`
`audio_features.py` has `TimeDomainFeatures` (AE/RMS/ZCR) and `FrequencyDomainFeatures`
(FFT/STFT/mel/MFCC/BER/spectral-centroid/bandwidth) on librosa/scipy. Strategy pattern
(`strategy/`): each `AFTypes` value → a strategy class extending `BaseStrategy`; dispatch is in
`AFStrategyFactory.create_strategy`. All imports here consistently use `src.utils.audio_features.*`
(the older `src.audio_features.*` bug is gone). To add a feature: add an `AFTypes` member, a
strategy subclass, and a factory branch.

### Schema model — `src/model_schema/model_schema_types.py`
Architecture is data, not code: `ModelSchema` = a list of `LayerSchema` (`type`/`units`/
`activation`/`regularizer`) + `optimizer` + `loss`. All choices are enums (`LayerType`,
`ActivationType`, `RegularizerType`, `OptimizerType`, `LossType`). `ExperimentDetails`
(`src/experiment/experiment_types.py`) holds *sequences* of these plus `units_range` — the search
space `main.py` declares and `ModeTuner` resolves down to concrete schemas one `ExperimentStep` at
a time.

### Shared utils — `src/utils/`
`files.py` (`Files`: path joins, `create_folder`, `get_only_files` — skips `.DS_Store`),
`wav_files.py` (`WavFiles`: librosa read / soundfile write), `logger/` (`Logger` implementing
`ILogger`, colorized stdout via `termcolor` — the only logging mechanism).

## Conventions

- **Interface + impl + types per component**, DI by constructor against the interface: an `I*`
  ABC, a concrete impl taking an injected `ILogger`, and (where needed) a `*_types.py` of
  enums/schemas.
- **Naming quirk:** classes/files are spelled `Mode*` (not `Model*`) — `ModeBuilder`, `ModeTrainer`,
  `ModeValidator`, `ModeTuner`, and the file `mode_validator_interfcace.py` (sic). Match existing
  spelling.
- **Indentation is inconsistent by subsystem**: `src/data_set/`, `src/utils/audio_features/`, and
  `src/assets_service/` use **2-space**; the newer experiment/model/database layers use
  **4-space**. Match the file you're editing.
- Changing `src/definitions.py:labels` (or `DURATION`) changes dataset/model fingerprints —
  existing DB experiments and cached `assets/` dirs for the old label set are simply orphaned, not
  migrated.
