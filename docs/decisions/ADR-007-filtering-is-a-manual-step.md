# ADR-007 — Filtering is a manual step between two runs

- **Date:** 2026-10-16
- **Status:** superseded by ADR-008 (filtering now also runs inside `prepare()` when an exported model exists)
- **Deciders:** maintainer (volodymyr)
- **Ticket:** 4

## Context
The pipeline filtered the cooked data set before every experiment step, using the in-memory best model of that
experiment (`InMemoryModelLoader`). The maintainer found it not effective in practice and `SKIP_FILTER` had switched
it off. `filter-data-set.py` (ticket 2) filters once with a chosen exported model. `DataSetCooker.prepare()` deleted
and re-split the data set at the start of every experiment, so that result could not survive into a run.

## Decision
`main.py` never filters. The flow is: run on the unfiltered data set, run `filter-data-set.py`, run again.
The `__filtered__` marker written by the script is the only signal the pipeline reads: `prepare()` keeps a marked
data set, removes its augmented files (names starting with an `ArgumentationTypes` value) and augments again, and
regenerates the shared validation records when they are older than the marker. Without the marker the data set is
re-split as before and not augmented, because augmenting mislabeled fragments multiplies the noise.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Keep per-step filtering behind `SKIP_FILTER` | not effective in practice (maintainer); two filtering paths |
| Write the filtered result to a separate folder the pipeline reads | the maintainer chose in-place filtering for ticket 2 |
| Track "filtered" in the experiment identity so identical settings can run twice | needs a schema change (rule P4) and is larger than the ticket; the maintainer changes settings or clears the row instead |

## Consequences
- Positive: one filtering path, chosen model named on the command line; the unfiltered and the filtered run are separate, visible runs.
- Accepted cost: a filtered data set stays until its folder is deleted; the shared validation records change after filtering, so later experiments are validated on different records than earlier ones.
- Accepted cost: the unfiltered run is recorded with the augmentation types it was asked for although none was applied, so run 1 should use `[nothing]` only; augmentation shares one folder, so `export-best-model.py` retrains on the last augmentation left there; augmented copies can leak into the validation split.
- Follow-ups: ticket 3 changes record generation (keep flag, 30 % rule) and should keep records per data set; separate tickets for the restorer and the validation-split leak.

## Revisit when
Identical settings must be run twice with both results kept in Postgres.
