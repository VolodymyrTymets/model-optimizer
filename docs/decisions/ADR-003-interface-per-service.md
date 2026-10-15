# ADR-003 — One ABC interface per service, injected by constructor

- **Date:** 2026-10-15
- **Status:** accepted
- **Deciders:** maintainer (retroactive)
- **Ticket:** none

_Recorded retroactively on 2026-10-15 from the code during `/setup`; not re-litigated._

## Context
Training, validation, tuning and feature extraction are swapped per experiment.

## Decision
Each service has `<name>_interface.py` with `I<Name>(ABC)` and an implementation beside it; dependencies are passed through the constructor. Audio features use one strategy per type behind `IAFStrategy` and `AFStrategyFactory`.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Plain functions and modules | not recorded; the code uses classes consistently |

## Consequences
- Positive: strategies and trainers are replaceable per experiment.
- Accepted cost: boilerplate for single-implementation services.

## Revisit when
A service never gets a second implementation.
