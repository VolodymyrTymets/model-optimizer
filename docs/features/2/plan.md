# 2 — implementation plan

<!-- A human writes this line to approve. Until it is here, guard-plan-approval blocks writes to code. -->
Approved-by: volodymyr · 2026-10-16

Pattern followed: `export-best-model.py` (thin entry script calling one service) plus the existing
`LocalModelLoader` / `DataSetFilter` in `src/data_set/utils/data_set_filter.py` (loader strategy chosen by the
`localModel` flag, ADR-003 shape: collaborators injected through the constructor). Path logic stays in
`src/assets_service` (rule P2/P5). No second pattern is introduced, so no ADR.

## Contract changes
- boundary: new CLI `.venv/bin/python filter-data-set.py <model_name> [--dry-run]`. Additive.
- code: `IAssetsService` gains one abstract method (the only implementer is `AssetsService`);
  `DataSetFilter.__init__` gains optional `model_name` and `dry_run`, and `experiment_id` becomes optional.
  Existing `DataSetCooker` call stays valid unchanged.
- data: moves files inside `assets/<DATA_SET_NAME>_<fingerprint>` and creates `__filtered__` there. Not reversible
  by the script (maintainer chose in place). No schema, no DB.
- generated output: none.

## Requirements (ordered, each independently verifiable)

### R1 — LocalModelLoader loads a named SavedModel (S)
- files: `src/data_set/utils/data_set_filter.py` (change: `LocalModelLoader.__init__(..., model_name)`; path =
  `join(*assets_service.get_models_path(model_name))`; load with `tf.saved_model.load`; `get_model` raises
  `FileNotFoundError` when the folder is absent; `_get_af_strategy` re-raises instead of logging and returning
  `None`; checks the model's `duration` from `parse_settings` equals `DURATION`)
- layer: service (data_set)
- test: none exists in `src/` (rule P6); verified by run
- executed how: every scratchpad script in R1-R3 runs as `PYTHONPATH=<repo> <repo>/.venv/bin/python <scratch>/x.py`
  with the working directory set to `<scratch>` (`ASSETS_PATH`
  is the relative `'assets'`; from the repo root the loader would open the real `assets/models/`, rule P5), and
  first prints `os.path.abspath(ASSETS_PATH)` and asserts it is under the scratchpad, before loading anything. That
  line is kept as evidence. The script exports a tiny Keras model with `ModelExporter.export_model` into
  `<scratch>/assets/models/model-<labels>-<duration>-<af>`, then loads it through `LocalModelLoader` and prints the
  settings read back (AC1, AC6). First step of the run: confirm that `tf.keras.models.load_model` really fails on
  that folder (the claim in the spec) and record the error text.
- risk: the Keras 3 claim is unverified until that run

### R2 — Model name to data set path, no DB (S)
- files: `src/assets_service/assets_service_interface.py` (new abstract
  `get_data_set_path_for_model(model_name) -> str`), `src/assets_service/assets_service.py` (implement: strip
  `model-`, `rsplit('-', 2)` into labels / duration / af type, return `join(ASSETS_PATH, f'{DATA_SET_NAME}_{labels}-{duration}')`,
  the same string `_get_data_set_fingerprint` builds; `ValueError` on a name that does not parse)
- layer: service (assets_service)
- executed how: scratchpad one-liner feeds the example name `model-breath_noise_stimulation-0.5-mfcc` under the
  sound task and prints the path; compares it with the literal `assets/data_set_sound_breath_noise_stimulation-0.5`
  (derived by hand from the format in `get_data_set_path`; `_get_data_set_fingerprint` needs the DB). A malformed
  name shows the `ValueError`. Also in this file: `IAssetsService.get_models_path` becomes
  `get_models_path(self, model_name: Optional[str] = None) -> tuple[str, str]` so R1's call matches the interface.
- risk: `AssetsService.__init__` builds `ExperimentModelService` -> `DBClient()`; the engine is lazy, so no Postgres
  is expected. If a connection is attempted, make the script construct a path-only helper instead (see Risks).

### R3 — DataSetFilter: model name, dry run, count-only logs (M)
- files: `src/data_set/utils/data_set_filter.py` (change: ctor takes `model_name: Optional[str] = None`,
  `dry_run: bool = False`, `experiment_id: Optional[int] = None`; passes `model_name` to `LocalModelLoader`;
  in `filter`, replace the per-file `moving <from> to <to>` log with a counter keyed by (set, from, to) logged once at
  the end; with `dry_run` skip `create_folder`, `shutil.move` and `finish()`; before the loop, check that every
  `<set>/<label>` folder exists and raise naming the set and label if one is missing, so a half-moved data set
  cannot happen; `test` is filtered like `train`, no `except_sets`, maintainer decision)
- layer: service (data_set)
- executed how: synthetic data set under `<scratch>/assets/data_set_<...>-<fingerprint>/{train,test}/<label>/*.wav`,
  generated from sine tones; the tiny model from R1 is untrained, so the check compares the actual moves with the
  model's own predictions (`ModelResultParser.parse`) per fragment. Runs: (0) a tree with one `<set>/<label>` folder
  missing: the error names set and label, listing before and after identical (nothing moved); (a) dry run on a full
  tree: listing (paths, sizes, mtimes) identical before and after, no marker, log lists the planned moves (AC4, AC7); (b) real run: each mismatching fragment moved to its predicted
  label, `keep_*.wav` and `time_shift_*.wav` fixtures untouched (AC3), marker present; (c) second run exits with
  the "already filtered" line (AC5). Log shows each move with its file name (AC7, changed from counts only at the maintainer's request).
- risk: AC8 is verified by reasoning, not by a run: `SKIP_FILTER = True` makes `DataSetCooker._filter_data_set`
  return before `filter()`. The in-memory path (needs Postgres + an experiment) goes under "Not verified" in the PR

### R4 — Entry script (S)
- files: `filter-data-set.py` (new, repo root): `argparse` with `model_name` and `--dry-run`; compares strings, not
  parsed lists (a label may contain `_`): `'_'.join(sorted(labels))` against the labels part of the name and
  `str(DURATION)` against the duration part, `ValueError` naming both (AC2; the name carries no task, so a
  wrong-task model is caught only through this labels/duration mismatch, which works for sound vs emg today); checks the data set folder exists;
  uses `AssetsService(experiment_id=None)` (path methods only, no DB query);
  builds the `AFStrategy` from the parsed `af_type` via `AFStrategyFactory`; runs `DataSetFilter(..., localModel=True,
  model_name=..., dry_run=...)`; prints the result like `main.py` does
- layer: entry script
- executed how: from the scratchpad dir, `cd <scratch> && /…/.venv/bin/python /…/filter-data-set.py <name>`, so the
  relative `ASSETS_PATH = 'assets'` resolves to the synthetic tree and the real `assets/` is never opened. Runs the
  R3 scenarios end to end, plus: wrong-task name (AC2), absent model folder (edge case), absent data set (edge case).
- risk: none

### R5 — Docs (S)
- files: `docs/ARCHITECTURE.md` (entry-script row, flow 4 "Filter by model", `__filtered__` marker landmine),
  `docs/RUNBOOK.md` (command + "filtering moves files, no undo"); both also note that `test` is filtered, so
  validation records inherit the model's labels (goal 1 caveat), `docs/DOMAIN_GLOSSARY.md` (Filtering, Keep flag,
  Model fingerprint), `docs/features/2/spec.md` (tick ACs)
- executed how: re-read the changed lines against the code

## Docs to update in this PR
- [ ] docs/features/2/spec.md · [ ] docs/ARCHITECTURE.md · [ ] docs/RUNBOOK.md · [ ] docs/DOMAIN_GLOSSARY.md
- ADR: none (existing loader/filter shape reused). CLAUDE.md untouched.

## Risks
| Risk | Impact | Cheapest way to find out early |
|------|--------|-------------------------------|
| R-1: `DataSetCooker.prepare()` calls `remove_data_set()` and re-splits, so the next `main.py` run deletes the in-place result | the script works but its output is lost; ticket 4 cannot deliver "second run uses the filtered set" | Read-only check done: `prepare` removes the folder first. Hand to ticket 4 explicitly; state in the PR body and the RUNBOOK. Not fixed here (out of scope, rule B4) |
| R-2: irreversible moves on a data set chosen by name; a wrong model corrupts labels | goal 1 and 2 | `--dry-run` first (documented); AC2 refuses a model from another task; counts only logged |
| R-3: Keras 3 / TF 2.20 loading of the exported SavedModel, and `AssetsService` touching the DB at construction | R1/R2 change | first minutes of R1 and R2 runs, both in the scratchpad |
| R-4: `filter` also runs on `test`, and validation records are built from `test`, so the model relabels held-out fragments and can inflate validation accuracy (goal 1) | misleading results | Decided by the maintainer 2026-10-16: do not exclude `test`. Kept as existing behaviour; named under "Not covered" in the PR body |

## Assumptions
- Work happens on branch `feat/2-filter-data-set` (rule A1), created at the start of `/implement`.
- The script is run by the maintainer on the real `assets/`; the agent only ever runs it on a synthetic tree (P5, G1).
- The `time_shift` / `pitch_shift` / `time_stretch` / `normalization` / `keep` skips in `filter` stay as they are.
- Model name format is `model-<labels joined by _>-<duration>-<af_type>`, from `AssetsService.get_models_path`.
- Python 3.9 `.venv`; no new dependency (rule C2). Complexity: S, S, M, S, S.
