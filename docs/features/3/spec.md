# 3 — Adjust DataSetRecordGenerator

Source: "Scope Breakdown" doc, task 3 ("We have special flag in data_set records name (keep). If a record has it in
its name it is definitely placed in the right label folder, so we skip it from filtering. Now we need to make sure
that a record with this flag is part of the generated records. At least 30% of generated records should include
those records.").

## Problem
A fragment with `keep` in its file name is known to sit in the right label folder, and `DataSetFilter` never moves it
(`data_set_filter.py:106`). `DataSetRecordGenerator.generate_test_record` ignores the flag: it draws fragments of the
`test` set at random, so the validation records (`assets/validation-records`) hold keep fragments only by chance, and
the trusted fragments may not appear in them at all. Record accuracy (`record_acc`, goal 1) is then measured mostly on
fragments whose label came from a model.

## Goal / business value
Goal 1 (held-out accuracy): the validation records contain a guaranteed share of fragments whose label is certain, so
`record_acc` is anchored on trusted ground truth. Goal 2: the share is a named setting, not an accident of `random`.

## Scope
- in: `generate_test_record` places keep-flagged `test` fragments into each generated record so that at least 30% of
  the fragments in it are keep-flagged (when enough exist); the share is a named tunable in `src/definitions.py`;
  a shortage is logged, never silent; a pytest test of the 30% minimum (adds pytest, approved by the maintainer);
  docs.
- **out (explicit):**
  - changing `DataSetFilter`, `DataSetSplitter` or the keep file naming (AC1 already holds, see below)
  - regenerating existing `assets/validation-records` automatically: records are regenerated only by the existing
    `_records_outdated()` rule (`__filtered__` newer than the folder) or when the maintainer deletes the folder
  - one records folder per data set / task (ADR-008 follow-up), the no-op `_remove_by_index`, the empty-pool
    `IndexError` in `_get_random`
  - reading, listing or editing `assets/` during development (rule P5): tests and checks use synthetic wav files

## Acceptance criteria
- [x] AC1 Given a fragment with `keep` in its name, when the data set is filtered, then it is not moved. Already true
      (`data_set_filter.py:106`, ticket 2); confirmed by reading, not changed.
- [x] AC2 Given the `test` set holds keep-flagged fragments, when records are generated, then every generated record
      includes at least one of them.
- [x] AC3 Given enough keep-flagged fragments, when records are generated, then at least 30% of the fragments in the
      generated records are keep-flagged (fragments overall; the same quota applies to every record).
- [x] AC4 A pytest test covers the 30% minimum, run with `.venv/bin/python -m pytest`.
- [x] AC5 Given too few or no keep-flagged fragments, when records are generated, then generation continues with
      all that exist and `Logger` records the shortfall (no error).

## Edge cases
| Case | Expected behaviour | Decided by |
|------|--------------------|-----------|
| No keep fragment in `test` (first run) | records as today; one warning line | maintainer, 2026-10-16 |
| Keep fragments only in some labels | quota capped by the slots those labels can fill; warning with the shortfall | maintainer (use all, warn) |
| Fewer keep fragments than slots of a label | picked with replacement, as every other pick today | assumption |
| 30% of a record's fragment count is not whole | rounded up | assumption |
| Keep fragments in `train` | ignored: records are built from `test` only (`except_sets=['train']`) | existing behaviour |

## Open questions
| # | Question | My assumed answer | Needs a human? |
|---|----------|-------------------|----------------|
| 1 | Unit of "30%" | fragments overall, enforced per record | answered 2026-10-16 |
| 2 | Shortage | use all, log a warning | answered 2026-10-16 |
| 3 | Test form | pytest, pinned in a new `requirements-dev.txt` (dev tools stay out of the runtime list); CLAUDE.md command map updated | answered 2026-10-16; the CLAUDE.md / requirements edits are ask-first paths |
| 4 | Should existing records be regenerated automatically? | no, out of scope; documented | no |
