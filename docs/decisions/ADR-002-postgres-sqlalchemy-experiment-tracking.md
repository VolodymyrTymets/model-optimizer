# ADR-002 — Track experiments in PostgreSQL through SQLAlchemy

- **Date:** 2026-10-15
- **Status:** accepted
- **Deciders:** maintainer (retroactive)
- **Ticket:** none

_Recorded retroactively on 2026-10-15 from the code during `/setup`; not re-litigated._

## Context
Grid and tuner runs produce many configurations and results; "best model" must be queryable (business goal 2).

## Decision
Experiment, step, model-schema and layer rows are stored in PostgreSQL via SQLAlchemy 2 declarative models (`src/database/schema.py`). Access goes through the `DBClient` singleton and `session_scope()`; tables are created with `create_all`, with no migration tool.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Files / CSV per run | not recorded; results would not be queryable across runs |

## Consequences
- Positive: `ModelRestorer.restore_best_step()` is a query.
- Accepted cost: a schema change has no migration path; the default URL with local credentials lives in `src/definitions.py`.

## Revisit when
The schema starts changing often (adopt a migration tool) or a second machine needs the history.
