# ADR-006 — Choose the task profile with MODEL_OPTIMIZER_TASK

- **Date:** 2026-10-16
- **Status:** accepted
- **Deciders:** maintainer (volodymyr)
- **Ticket:** 1

## Context
`src/definitions.py` held the settings of every signal type, and switching between sound and EMG meant commenting
and uncommenting blocks. A half-edited file silently pointed the pipeline at the wrong data set, database and
fingerprint, and the toggle itself showed up as commits (`75921cc`, `b2f2673`). Rule P2 said tunables live in that
one file.

## Decision
`src/definitions.py` stays the only module the rest of the code imports from. Task-specific values live in
`src/definitions_<task>.py` (`sound`, `emg`): `DEFAULT_DATABASE_URL`, `DATA_SET_NAME`, `VALIDATION_RECORDS_COUNT`,
`sr`, `DURATION`, `labels`, `labels_colors`. Everything shared (frame sizes, `ASSETS_PATH`, flags) is defined once in
`definitions.py`, which also derives `FRAGMENT_LENGTH`.

The profile is chosen by the environment variable `MODEL_OPTIMIZER_TASK`, from the shell or a repo-root `.env`
(loaded with `python-dotenv`; a shell value wins). Unset means `sound`; any other value raises `ValueError`. The
active name is exported as `TASK`.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Keep the commented blocks | silent wrong-task runs; the toggle is a commit |
| A package `src/definitions/` | no gain over flat modules, and it would change the import path's file layout for nothing |
| Fail when the variable is unset | breaks today's `python main.py` for sound; open to revisit |
| Hand-written `.env` parser | more code in the most-imported module than one pinned library |

## Consequences
- Positive: the task is a visible input to a run; the 24 importing modules are unchanged.
- Accepted cost: a new runtime dependency (`python-dotenv==1.2.1`, last release supporting Python 3.9). Forgetting
  the variable on an EMG run silently uses sound (database `model-optimizer-sound`, `data_set_sound`).
- Follow-ups: tickets 2 and 3 may need more per-task values; add them to the profile files, not to `definitions.py`.

## Reference data kept from the old file (not selectable profiles)
EMG-hands: `labels = ['noise', 'WristExtension', 'WristFlexion']`, colours blue / red / green; the optional extra
classes were `Supination`, `Rest`, `Pronation`, `HandOpen`, `HandClose` (yellow, black, purple, orange, pink).
Speech: `labels = ['noise', 'down', 'up', 'go', 'left', 'right', 'stop', 'yes', 'no']`, with colours noise blue,
down red, up green, left yellow, right black, go purple, stop orange, yes pink, no brown. The full old file is in
git at `b2f2673`.

## Revisit when
A third task is added, or a forgotten variable costs a wrong run.
