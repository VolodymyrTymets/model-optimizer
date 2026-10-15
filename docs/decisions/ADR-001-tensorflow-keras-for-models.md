# ADR-001 — Use TensorFlow/Keras for model training and export

- **Date:** 2026-10-15
- **Status:** accepted
- **Deciders:** maintainer (retroactive)
- **Ticket:** none

_Recorded retroactively on 2026-10-15 from the code during `/setup`; not re-litigated._

## Context
The pipeline trains many small classifiers and exports them for a device.

## Decision
Models are built, trained and exported with TensorFlow 2.20 / Keras 3.10 (`tf.keras`, SavedModel with a `get_settings` signature). `requirements-gpu.txt` adds `tensorflow-metal` for Apple GPUs.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| PyTorch | not evaluated in the code; no reason recorded |

## Consequences
- Positive: SavedModel export is a single call (`ModelExporter`).
- Accepted cost: TF, Keras, NumPy and Python 3.9 are pinned together; bumping any one is its own PR.

## Revisit when
The target device needs a runtime a TF SavedModel cannot satisfy.
