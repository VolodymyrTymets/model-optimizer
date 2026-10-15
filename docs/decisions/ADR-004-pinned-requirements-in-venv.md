# ADR-004 — Exact-pinned requirements files inside a project venv

- **Date:** 2026-10-15
- **Status:** accepted
- **Deciders:** maintainer (retroactive)
- **Ticket:** none

_Recorded retroactively on 2026-10-15 from the code during `/setup`; not re-litigated._

## Context
The TF / Keras / NumPy / Numba stack is version-sensitive and runs on a laptop with and without an Apple GPU.

## Decision
Dependencies are exact `==` pins in `requirements.txt` (CPU) and `requirements-gpu.txt` (adds `tensorflow-metal`), installed into `.venv` / `.venv-gpu`. There is no lockfile, no `pyproject.toml` and no `.python-version`.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| uv / poetry | not adopted; would be its own PR |

## Consequences
- Positive: reproducible versions with plain tooling.
- Accepted cost: the interpreter version (3.9) is implicit and the two requirement files can drift.

## Revisit when
A second contributor joins, or the interpreter changes.
