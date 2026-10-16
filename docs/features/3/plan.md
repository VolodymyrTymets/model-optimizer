# 3 — implementation plan

<!-- A human writes this line to approve. Until it is here, guard-plan-approval blocks writes to code. -->
Approved-by: volodymyr · 2026-10-16

Pattern followed: `src/data_set/utils/data_set_record_generator.py` itself (random picks per segment, composed in
`generate_test_record`), the `keet_prefix in file` check of `data_set_filter.py:106` and `data_set_splitter.py:21`,
and the tunables-in-`src/definitions.py` rule P2. No second pattern, so no new-pattern ADR. No test precedent exists
(rule P6): pytest layout follows `.claude/rules/testing-python.md`.

## Contract changes
- boundary: `generate_test_record` gains `keep_share: float = KEEP_RECORDS_SHARE`; its one caller
  (`DataSetCooker._generate_records`) is unchanged.
- data: `validation-records/test_*.wav` and `.annotation.json` keep their format. Content changes only for records
  generated after this PR; existing records are regenerated only by the cooker's existing rule (`_records_outdated()`: `__filtered__` newer than the
  folder) or when the maintainer deletes the folder (documented).
- config: new `KEEP_RECORDS_SHARE: float = 0.3` in `src/definitions.py` (shared by every task).
- dependencies: `pytest` and its transitive dependencies, all pinned with `==`, in a new `requirements-dev.txt`
  (python-tooling rule 5: dev tools never in the runtime list). `requirements.txt` and `requirements-gpu.txt` stay unchanged.
- generated output: none.

## Requirements (ordered, each independently verifiable)

### R1 — pytest harness (S)
- files: `requirements-dev.txt` (new: pytest, a version supporting 3.9 and 3.11 because ticket 5 bumps Python, plus
  every transitive dependency pinned; the set is read from a throwaway venv's `pip freeze` diff, not guessed);
  `pytest.ini` (new: `testpaths = tests`, `addopts = --strict-markers`); `tests/conftest.py` (new: sets
  `MODEL_OPTIMIZER_TASK=sound` before any `src` import so the shell/.env cannot change the task, rule 12);
  `CLAUDE.md` (change: `install` row adds `.venv/bin/pip install -r requirements-dev.txt`, rows `unit tests` and
  `affected tests` become `.venv/bin/python -m pytest`; note in P6); `.claude/rules/testing.md` (change: line 11 says
  there is no test suite). `requirements-dev.txt`, `CLAUDE.md` and `testing.md` are all ask-first paths
- layer: tooling
- test: `tests/test_smoke.py` is NOT added; the harness is proven by R2's test
- executed how: `.venv/bin/pip install -r requirements-dev.txt`, then `.venv/bin/python -m pytest --collect-only`
- risk: `.venv-gpu` gets pytest only by `.venv-gpu/bin/pip install -r requirements-dev.txt` (the file is shared; the
  pins must also resolve there, and the venvs are never mixed, S2); the GPU venv cannot be checked here, so it is listed under "Not verified"

### R2 — keep-aware record generation (M)
- files: `src/definitions.py` (change: `KEEP_RECORDS_SHARE`);
  `src/data_set/utils/data_set_record_generator.py` (change: pools hold `(signal, is_keep)` with
  `is_keep = keet_prefix in file`; segments of a record are planned first (initial noise, then label + noise per
  label); `quota = ceil(keep_share * total_slots)`; new pure `_allocate_keep_slots(capacities, quota)` spreads keep
  slots round-robin over the segments whose pool holds a keep fragment, capped by capacity; each segment draws its
  keep slots from the keep subset, the rest from the whole pool, then shuffles; a shortfall is logged with
  `self.logger.log(..., color='yellow')` (`Logger` has no warning level and prints, P3), counts only, no file names (G1, D8))
- layer: service (data_set)
- test: `tests/test_data_set_record_generator.py` — synthetic wavs in `tmp_path` where keep fragments have a distinct
  amplitude and `labels` is monkeypatched on the module; seeded `random`. Cases: (a) share met for several record
  counts and keep-pool sizes (parametrized), measured on the written wav; (b) every record has ≥1 keep fragment;
  (c) no keep fragment: records still written, warning printed (asserted with `capsys`, `Logger` uses `print`);
  (d) keep only in the noise pool: quota capped, warning printed; (e) `_allocate_keep_slots` spreads and caps. Written first and seen failing (the old generator gives ~0 keep
  fragments), then passing.
- executed how: `.venv/bin/python -m pytest tests -q`; plus a scratch run (cwd = scratch dir, synthetic `assets/`,
  `ExperimentModelService.get_data_set_details` patched; the harness is written fresh in the scratchpad, the ticket 4
  one was never committed) calling
  `DataSetCooker._generate_records`, which asserts the keep share in the written records
- risk: see Risks

### R3 — Docs (S)
- files: `docs/ARCHITECTURE.md` (flow 1 / constraints: records carry ≥30% keep fragments; records follow the new rule after the next
  filtering or after `assets/validation-records` is deleted); `docs/DOMAIN_GLOSSARY.md` (keep flag row: placed in records, `KEEP_RECORDS_SHARE`);
  `docs/decisions/ADR-004` (amend: `requirements-dev.txt` is a third, dev-only file); `docs/ARCHITECTURE.md` also
  gets the requirements row, `tests/` and `pytest.ini` rows; `docs/RUNBOOK.md` (dev install in "First run", tests command; delete `assets/validation-records` to regenerate, also regenerated after the next filtering);
  `docs/features/3/spec.md` (ACs checked off)
- executed how: re-read against the final code

## Docs to update in this PR
- [ ] docs/features/3/spec.md (acceptance criteria checked off)
- [ ] docs/ARCHITECTURE.md — constraint on records; test layout (`tests/`) in the repo layout table
- [ ] docs/DOMAIN_GLOSSARY.md — keep flag
- [ ] docs/RUNBOOK.md — "The repo has no tests" is no longer true
- [ ] .claude/rules/testing.md — stale "no test suite" line (in R1)
- [ ] ADR-004 amended for `requirements-dev.txt`; no ADR for the generator (no new pattern)

## Risks
| Risk | Impact | Cheapest way to find out early |
|------|--------|-------------------------------|
| Picking with replacement (the removal helpers are no-ops) means a small keep pool repeats the same fragment in a record | validation looks less diverse | test case with a keep pool of 1; report it, do not fix here (C1) |
| Importing `src.data_set...` pulls `src.definitions` (reads `.env`, task) and librosa at import | flaky tests across machines | `tests/conftest.py` pins the task; R1 collect-only run on a clean shell |
| Existing records stay old, so a run right after merge shows no change | confusion | documented in RUNBOOK and ARCHITECTURE; PR body says so |
| `record_acc` before and after this PR measures different record mixes, so a Postgres "best step" query across both mixes them | goal 2 | compare only experiments run after the change; named in the PR body under "NOT covered" |
| Guaranteed keep fragments are chosen by the maintainer, possibly easy ones | `record_acc` may rise | named in the PR body; share is one constant |

## Assumptions
- "30% of generated records" means 30% of the fragments in them, enforced in every record (maintainer, 2026-10-16).
- Shortage: use all available, warn (maintainer, 2026-10-16). Rounding up of the quota.
- pytest goes in `requirements-dev.txt`, not `requirements.txt`: the maintainer's "add pytest" is read as satisfied by this (python-tooling rule 5). Say so if you want it in `requirements.txt`.
- The maintainer commits or stashes the unrelated `.claude/rules/protected-paths.md` edit before `/implement` branches (A4, C1).
- AC1 needs no code change (filter and splitter already honour `keep`).
- Unconfirmed until R2 runs: the scratch harness builds a cooker without Postgres, and `WavFiles.read` keeps the synthetic `sr`.
- Not verified in this PR: `.venv-gpu`; a real `main.py` run (Postgres, recordings, training); `assets/` is never touched (P5).
