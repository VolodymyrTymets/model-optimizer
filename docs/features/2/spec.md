# 2 — Filtering script

Source: "Scope Breakdown" doc, task 2 ("Need to add filter script. See export-best-model.py as an example. It should
use DataSetFilter to filter data_set by model. Check LocalModelLoader, it should load model from
assets/models/<model_fingerprint>, e.g. model-breath_noise_stimulation-0.5-mfcc").

## Problem
`DataSetFilter` (`src/data_set/utils/data_set_filter.py`) is only reachable from inside the grid pipeline, where it
filters on every step with the in-memory best model (not effective in practice, per the doc). There is no way to
filter a cooked data set once, by hand, with a chosen exported model. `LocalModelLoader` was meant for that but is
unused and does not match the exported layout: it looks for `assets/models/model_<duration>` (the exporter writes
`assets/models/model-<labels>-<duration>-<af_type>`) and loads it with `tf.keras.models.load_model`, which Keras 3
does not accept for a SavedModel folder (the exporter's own `load_model` uses `tf.saved_model.load`).

## Goal / business value
Goal 1 (held-out accuracy): mislabeled fragments in the cooked data set are moved to the label the model predicts, so
the next run trains on cleaner data. Goal 2: the model used is named on the command line, not implied.
Prerequisite for ticket 4 (remove per-step filtering from `main.py`).

## Scope
- in: new entry script `filter-data-set.py` (shape of `export-best-model.py`); `LocalModelLoader` loads a named
  model folder; `DataSetFilter` accepts a model name and a dry-run flag; one `AssetsService` method that maps a model
  name to its data set folder (no DB needed); docs.
- **out (explicit):**
  - removing or changing filtering in `main.py` / `DataSetCooker` (ticket 4); `DataSetCooker` call sites keep today's
    behaviour (in-memory loader)
  - keep-flag handling in `DataSetRecordGenerator` and the 30 % rule (ticket 3); the filter already skips `keep`
    files and augmented files, unchanged
  - a `--force` to re-filter, undo of a filtering run, filtering by TFLite model
  - reading, listing or editing `assets/` during development (rule P5): all runs use a synthetic data set

## Acceptance criteria
- [x] AC1 Given `python filter-data-set.py <model_name>` with `assets/models/<model_name>` present, then the script
  loads that SavedModel (not `model_<duration>`) and filters `assets/<DATA_SET_NAME>_<labels>-<duration>`.
- [x] AC2 Given the model's labels or duration do not match the active definitions (the model name carries no task, so a wrong task is caught only this way), then it raises
  `ValueError` naming both values before any file is moved.
- [x] AC3 Given a fragment whose predicted label differs from its folder, then it is moved to the predicted label's
  folder; files with `keep` in the name and augmented files are never moved.
- [x] AC4 Given `--dry-run`, then nothing is moved and no `__filtered__` marker is written; the log lists what would move.
- [x] AC5 Given the data set is already marked `__filtered__`, then the script logs and exits without moving.
- [x] AC6 Given the model cannot be loaded or its settings cannot be read, then it raises; it never filters with a
  default feature strategy (rule D5).
- [x] AC7 The log lists every move as `<set>/<label>/<file> -> <set>/<new label>` plus counts per (set, from-label, to-label). Changed on 2026-10-16 at the maintainer's request (was counts only). The output stays on the maintainer's terminal; it must not be pasted into prompts, issues or PRs (rule G1).
- [x] AC9 Given a `<set>/<label>` folder is missing, then the script raises naming both and moves nothing.
- [x] AC10 Given a planned move would overwrite an existing file, then it raises `FileExistsError` and nothing is moved.
- [x] AC8 `main.py` is unaffected while `SKIP_FILTER = True`: it never reaches `filter()`. With the flag off, the in-memory loader runs the shared `filter()` changes (folder pre-check, collision guard, count-only logs). Checked by reasoning only, not run (needs Postgres and an experiment).

## Edge cases
| Case | Expected behaviour | Decided by |
|------|--------------------|-----------|
| model name missing or folder absent | `FileNotFoundError` with the expected path | assumption |
| model built with another feature type than the strategy default | strategy created from the model name's `af_type` | code (`_get_af_strategy`) |
| cooked data set folder absent (first run not done) | raise, tell the user to run `main.py` first | assumption |
| a `<set>/<label>` folder missing inside the data set | raise naming set and label before any file is moved | analyze gate (no half-moved data set) |
| fragment shorter than `FRAGMENT_LENGTH` | skipped, as today | code |

## Open questions
| # | Question | My assumed answer | Needs a human? |
|---|----------|-------------------|----------------|
| 1 | In place or separate copy? | In place on the cooked `data_set_<fingerprint>` (answered by the maintainer). The doc's ticket 4 must then deal with `DataSetCooker.prepare()` deleting that folder at the start of every run | answered; risk R-1 in plan |
| 2 | How is the model chosen? | one positional CLI argument, the folder name under `assets/models/` | no |
| 3 | Re-run for a second model? | blocked by the existing `__filtered__` marker; no `--force` now | no |
| 4 | Filter the `test` set too? | Yes, both `train` and `test` (answered by the maintainer, 2026-10-16) | answered |
