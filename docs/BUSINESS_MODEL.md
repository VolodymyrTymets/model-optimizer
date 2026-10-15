# Business model

> Why this product exists. This file is the **tie-breaker** the agent uses when two technically valid options collide.

## What this product is
Automated search and tuning of small TensorFlow/Keras classifiers for EMG and audio signals (e.g. noise vs stimulation), producing a model that is deployed to a device. Used by the maintainer's research/engineering work.

## Ranked goals
1. **Model accuracy on held-out validation records** — the number that decides whether a model is any good.
2. **Reproducible experiments tracked in Postgres** — every run's settings and results are recoverable, so "best model" is a query, not a memory.
3. **Small, fast, exportable models** — the target is a device, so size and inference speed matter once accuracy is acceptable.

## Non-negotiables (never trade these for speed)
- No recorded user data (EMG / audio / MRI under `assets/`) leaves this machine: not into prompts, logs, issues or PRs.
- No production host or credential; local Postgres only.

## Explicitly out of scope (this phase)
- <not stated during setup — maintainer to fill in; until then nothing is declared out of scope>

## Judgement examples
| Situation | Decide this way | Because |
|-----------|-----------------|---------|
| Faster training vs held-out accuracy | keep accuracy | goal 1 outranks speed |
| Easier code vs experiment still reproducible | keep reproducible | goal 2 |
| Bigger model, +0.3 % accuracy | prefer it only if it still exports and fits the device | goals 1 then 3 |

## Soft values
Pragmatic research code: clarity of the pipeline matters more than polish. Do not rename or reshuffle modules without a reason tied to a goal above.
