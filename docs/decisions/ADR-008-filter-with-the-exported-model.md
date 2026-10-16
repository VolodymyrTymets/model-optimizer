# ADR-008 — Filter inside prepare() with the exported model

- **Date:** 2026-10-16
- **Status:** accepted
- **Deciders:** maintainer (volodymyr)
- **Ticket:** 4 (revision 1, feedback on PR #14)
- **Supersedes:** ADR-007

## Context
ADR-007 made filtering a manual step between two runs. Running `main.py` showed the maintainer a freshly cooked data
set that was not filtered (`Data set is not filtered. Skipping argumentation`), and the maintainer wants the pipeline
to filter right after cooking, without a separate script run.

## Decision
`DataSetCooker.prepare()` filters once after it cooks (re-splits) a data set, with the model in
`assets/models/model-<labels>-<duration>-<af_type>` if that folder exists, using the same `DataSetFilter` and
`LocalModelLoader` as `filter-data-set.py`. No such model (first run): no filtering, no augmentation. A data set that
already carries `__filtered__` is kept and not filtered again. Augmentation, the `__filtered__` gate and the
regeneration of the validation records work as in ADR-007. That folder is written only by `export-best-model.py`,
so the flow is `main.py`, `export-best-model.py`, `main.py`. The model must load and match the duration, otherwise
the run stops (rule D5).

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Load the best finished experiment's model from `assets/experiment-<id>/model` through a DB lookup | would make a plain grid run filter by itself, but needs a new query and a choice of "best"; the maintainer picked the existing export |
| Keep the manual script as the only way (ADR-007) | the maintainer expects the pipeline to filter after cooking |
| Filter per step with the in-memory best model | removed in ticket 4: not effective in practice |

## Consequences
- Positive: one pipeline path; the filter model is a visible, named folder; `filter-data-set.py` stays for a dry run or a manual pass.
- Accepted cost: a plain grid run never filters (the model appears only after `export-best-model.py`); the model must match the experiment's feature type exactly (maintainer choice), so with the one exported model only that feature type filters, experiments for other feature types that precede it in the loop run unfiltered and record augmentation that was not applied, and run 2 should list the exported feature type first; `train-rare-model.py` opts out with `with_filter=False`; the model was trained on an earlier split and has seen part of the new `test` set, which favours its own labels; the shared validation records are regenerated (randomly) after each filtering; a failing model stops the whole run.
- Follow-ups: ticket 3 (records per data set, keep flag); the restorer and validation-split issues listed in ADR-007.

## Revisit when
The best finished experiment's model should be used without running `export-best-model.py`.
