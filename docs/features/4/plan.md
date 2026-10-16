# 4 — implementation plan

<!-- A human writes this line to approve. Until it is here, guard-plan-approval blocks writes to code. -->
Approved-by: volodymyr · 2026-10-16

Pattern followed: `src/data_set/data_set_cooker.py` itself (private `_`-prefixed steps that return early with a
`Logger('DataSet')` line, composed in `prepare`) and the `__filtered__` marker that `LocalModelLoader` already writes
and reads in `src/data_set/utils/data_set_filter.py`. No second pattern, so no new-pattern ADR; one ADR records the
settled decision "filtering is a manual step between two runs" (rule E1).

## Contract changes
- boundary: none for the CLI. `DataSetCooker.step_prepare` is removed (one caller, `ExperimentStep.run`);
  `DataSetCooker.prepare` keeps its signature; `DataSetFilter.__init__` loses `localModel` and `experiment_id`
  (callers: the cooker, which stops using it, and `filter-data-set.py`).
- data: a filtered `assets/data_set_<fingerprint>` is no longer deleted by `prepare()`. Unfiltered ones are
  handled as today. No schema, no DB.
- config: `SKIP_FILTER` is removed from `src/definitions.py` (only read in the cooker).
- generated output: none.

## Requirements (ordered, each independently verifiable)

### R1 — Cooker reuses a filtered data set (M)
- files: `src/data_set/utils/data_set_filter.py` (change: module constant `FILTERED_MARKER = '__filtered__'`, used by
  `LocalModelLoader`); `src/data_set/data_set_cooker.py` (change: `_is_filtered()` = marker exists in the data set
  path; `prepare()` calls `remove_data_set()` + `_split_data_set()` only when not filtered; `_argument_data_set`
  gates on `_is_filtered()` and first calls the new `DataSetTransformer.remove_argumentation()`;
  `_generate_records(is_need_to_regenerate=self._records_outdated())` where `_records_outdated()` = `_is_filtered()` and
  records folder exists and `getmtime(marker) > getmtime(records)`: on an unfiltered data set it is `False` without
  reading any mtime, and a missing folder is just generated);
  `src/data_set/utils/data_set_transformer.py` (new method `remove_argumentation(except_sets, except_labels)`: deletes
  files whose name starts with a non-`nothing` `ArgumentationTypes` value in the same sets/labels `argument` writes
  to; `argument` itself is not changed: with nothing left, its per-type `continue` never fires)
- layer: service (data_set)
- test: none exists in `src/` (rule P6); verified by run
- executed how: scratch script, cwd = a fresh scratch dir whose `assets/` holds a synthetic raw data set
  (`assets/data_set_sound/{train,test}/<label>/*.wav`, sine tones); it asserts `abspath(ASSETS_PATH)` is under the
  scratchpad, patches `ExperimentModelService.get_data_set_details` to return a fixed `ExperimentDataSetDetails`
  (no Postgres needed). Command: `cd <scratch>/runN && MODEL_OPTIMIZER_TASK=sound PYTHONPATH=<repo> <repo>/.venv/bin/python ../check_4.py`
  (a shell value beats the repo `.env`); the script also asserts `DATA_SET_NAME == 'data_set_sound'`. Runs: (a) unfiltered: `prepare()` splits, listing shows no
  augmented file (AC1); (a2) unfiltered with an existing records folder: `prepare()` raises nothing and the records
  folder mtime is unchanged (AC1, AC4); (b) write `__filtered__`, move one fragment by hand, `prepare()` again with
  `[time_shift]`: the moved fragment is still in its new folder, time_shift files exist for train only, marker
  intact (AC2, AC3); (c) `prepare()` again: file count unchanged (AC3); (e) on one filtered set call `prepare()`
  with `[normalization]`, then `[normalization, time_stretch]`, then `[pitch_shift]`; after each print the file
  count per name prefix: only the requested types remain, base fragment count constant (AC3); (d) records older
  than the marker are regenerated, newer ones kept, shown by folder mtimes (AC4); (d2) records folder deleted with
  the marker present: `prepare()` generates it (AC4)
- risk: `DataSetFile` names / augmentation idempotency, see Risks

### R2 — Remove per-step filtering (S)
- files: `src/data_set/data_set_cooker.py` (remove `data_set_filter` member, `_filter_data_set`, `step_prepare`,
  the `SKIP_FILTER` import; `af_strategy` parameter stays unused so four callers are untouched);
  `src/experiment/experiment_step/experiment_step.py` (remove the `step_prepare` call, the `data_set_cooker` member
  and its import); `src/definitions.py` (remove `SKIP_FILTER`)
- layer: orchestration (experiment) and service (data_set)
- executed how: `grep -rn "step_prepare\|SKIP_FILTER" src *.py` is empty; `<repo>/.venv/bin/python -m py_compile <touched files>`; `<repo>/.venv/bin/python -c "import src.experiments.experiments, main"` succeeds; R1's scratch script re-run unchanged (AC5, AC7). The full
  `main.py` run needs Postgres, recordings and training and is listed under "Not verified".
- risk: none

### R3 — Remove the in-memory filter path (S)
- files: `src/data_set/utils/data_set_filter.py` (remove `InMemoryModelLoader`, the `localModel` and `experiment_id`
  parameters and the unused imports; `DataSetFilter` always uses `LocalModelLoader`; `model_name` required);
  `filter-data-set.py` (drop `localModel=True`)
- layer: service (data_set), entry script
- executed how: ticket 2's scratch checks re-run (dry run, real run, missing folder, collision, forced
  `PermissionError`, constructor guard now only `model_name is None`) and `filter-data-set.py` end to end on a
  synthetic tree (AC6)
- risk: none

### R4 — Docs and ADR (S)
- files: `docs/decisions/ADR-007-filtering-is-a-manual-step.md` (new), `docs/ARCHITECTURE.md` (flow 1 loses the
  filter step, flow 4 mentions the reuse, landmine "prepare deletes the data set" replaced), `docs/RUNBOOK.md` (the
  two-run procedure, clearing the experiment row, deleting a filtered folder to re-cook), `docs/features/4/spec.md`
  (tick the ACs)
- executed how: re-read the changed lines against the code

## Docs to update in this PR
- [x] spec.md · [x] ARCHITECTURE.md · [x] RUNBOOK.md · [x] ADR-007 · [x] glossary (augmentation wording)

## Risks
| Risk | Impact | Cheapest way to find out early |
|------|--------|-------------------------------|
| R-1: `DataSetTransformer.argument` is not safe on a reused data set (its "done" check skips the rest of a file after one done type, and it would re-augment augmented files) | wrong or doubled augmented files, recorded settings no longer describe the data (goal 2) | `remove_argumentation()` before `argument`; R1 run (e) counts files per prefix across three differing lists |
| R-2: a filtered folder now lives until the maintainer deletes it; changed raw recordings are not picked up | stale data set, goal 2 | documented in RUNBOOK; the folder name already changes with labels or duration |
| R-3: records are regenerated from a changed `test` set, and validation records are shared by all experiments | a later experiment is validated on different records than an earlier one | explicit in the RUNBOOK; AC4 limits regeneration to once per filtering |
| R-4: the full pipeline cannot run here (Postgres, recordings, training) | the wiring in `ExperimentStep` is verified by import and grep only | listed under "Not verified" in the PR |

## Assumptions
- Branch `feat/4-adjust-filtering` from `main` (which already contains ticket 2), created at the start of `/implement`.
- Python 3.9 `.venv`, no new dependency (rule C2). Complexity: M, S, S, S.
- The uncommitted local change to `.claude/rules/protected-paths.md` stays out of this PR.
