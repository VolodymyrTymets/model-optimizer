# 4 — Adjust filtering mechanism

Source: "Scope Breakdown" doc, task 4 ("Need to adjust filtering mechanism in main.py pipeline. Current: we filter
data set on each step on algorithm by best model, not effective in practice. Desirable: run first time without
data_set filtering, then filter by the filter script from task 2, then run second time with the filtered data_set.
Need to do: remove filtering from main pipeline; do not add it to pipeline; do not change pipeline").
Not to be confused with `docs/features/4.4` (TFLite export).

## Problem
`ExperimentStep.run` calls `DataSetCooker.step_prepare` before every step; `step_prepare` filters the data set with
the in-memory best model of the experiment (`InMemoryModelLoader`), then augments. `SKIP_FILTER = True` switches the
filter off, but the code path, the flag and the loader stay. Worse, the way the pipeline treats a data set makes the
wanted flow impossible: `DataSetCooker.prepare()` deletes and re-splits `assets/data_set_<fingerprint>` at the start of
every experiment, so a data set filtered by `filter-data-set.py` (ticket 2) is lost before the second run reads it.
Augmentation is also gated on the in-memory `is_filtered()`, which is false for a fresh experiment, so augmentation
does not run today either.

## Goal / business value
Goal 1 (held-out accuracy): train the second run on a data set cleaned by one deliberate, chosen-model filtering pass
instead of a noisy per-step one. Goal 2: the unfiltered run and the filtered run are separate, visible runs.

## Scope
- in: the cooker keeps and reuses a data set that carries `__filtered__` (written by the ticket 2 script) and
  re-cooks any other one as today; augmentation runs only on a filtered data set, gated by the marker; validation
  records are regenerated once after the data set was filtered; per-step filtering and its flag are removed
  (`step_prepare`, `SKIP_FILTER`, `InMemoryModelLoader`, the `localModel` switch); docs and an ADR.
- **out (explicit):**
  - adding filtering anywhere in `main.py` / `Experiments` (ticket: "do not add it to pipeline")
  - changing the experiment, tuner, trainer or validator logic; the schema, DB or `Experiment.is_finished()`
  - keep-flag handling and the 30 % rule in `DataSetRecordGenerator` (ticket 3)
  - a way to run identical settings twice: the maintainer starts the second run with different settings or by
    clearing the finished experiment row (decided 2026-10-16)
  - changes to `filter-data-set.py` behaviour; any `assets/` access during development (rule P5)

## Acceptance criteria
- [x] AC1 Given a data set without `__filtered__`, when `DataSetCooker.prepare()` runs, then it removes and re-splits
  it as today and does not augment it.
- [x] AC2 Given a data set with `__filtered__`, when `prepare()` runs, then no file of it is removed or rewritten by
  the split and the marker stays.
- [x] AC3 Given a filtered data set, then after `prepare()` its augmented files are exactly the requested types, each
  applied once: earlier augmented files are removed first (base fragments are never touched), so a run with
  `[normalization]`, then `[normalization, time_stretch]`, then `[pitch_shift]` leaves only that list's types, and
  repeating a call leaves the file count unchanged.
- [x] AC4 Given a filtered data set newer than the validation records, then the records are regenerated; given
  records newer than the marker, they are kept.
- [x] AC5 No path of `main.py` filters: no reference to `step_prepare`, `SKIP_FILTER`, `InMemoryModelLoader` or
  `localModel` remains in `src/` or the entry scripts.
- [x] AC6 `filter-data-set.py` still works (dry run and real run on a synthetic data set).
- [x] AC7 `ExperimentStep.run` no longer touches the data set on disk before importing it.

## Edge cases
| Case | Expected behaviour | Decided by |
|------|--------------------|-----------|
| Filtered data set exists but raw recordings or labels changed | labels/duration change the folder name, so a new one is cooked; changed raw recordings need the maintainer to delete the filtered folder (documented) | assumption |
| `EMULATE_MODE` is on | `prepare()` behaves as today (augment and records are skipped) | code |
| Validation records folder missing | generated, as today | code |
| Augmentation list differs between experiments on the same filtered data set (lists are cumulative and reset per feature type) | augmented files of the previous list are removed (recognised by a name starting with an `ArgumentationTypes` value), then the requested ones are made | analyze gate (goal 2: recorded settings must describe the data) |
| Second run with identical settings | skipped by `Experiments.run` (`is_finished()`); the maintainer changes settings or clears the row | maintainer |

## Open questions
| # | Question | My assumed answer | Needs a human? |
|---|----------|-------------------|----------------|
| 1 | How to start the second run | different settings or clearing the finished row, no code (answered by the maintainer) | answered |
| 2 | Augment on the first (unfiltered) run? | No: today it is skipped for the same reason (augmenting mislabeled data), so nothing changes | no |
| 3 | `DataSetCooker(af_strategy=...)` parameter becomes unused | kept, so four callers stay untouched | no |
