# ADR-005 — Export a float32 TFLite model with a settings sidecar beside the SavedModel

- **Date:** 2026-10-15
- **Status:** accepted
- **Deciders:** maintainer
- **Ticket:** 4.4

Amends ADR-001: its "Revisit when" condition (the device needs a runtime a TF SavedModel cannot satisfy) is met.
ADR-001 still holds for training and for the SavedModel export.

## Context
The model is deployed to a device that is not in this repo. A SavedModel is not a runnable on-device artefact; the
device needs a `.tflite` flatbuffer. The conversion used to be a manual step outside the repo.

## Decision
`ModelExporter.export_tflite_model` writes, next to the SavedModel in `<path>_<DURATION>_<af_type>/`:
- `model.tflite`: float32, builtin ops only, converted with `tf.lite.TFLiteConverter.from_keras_model`.
  Input is the raw Keras input (float32, batch dimension, no reshape). Output is the probabilities only, with no
  `class_names`, unlike the SavedModel `__call__`.
- `model_settings.json`: `labels`, `af_type`, `duration`, `input_shape`, because the flatbuffer carries none of them.

Both files are written under a temporary name and renamed only when both exist. On failure the error is logged,
any `model.tflite` / `model_settings.json` in that folder (including ones from a previous export) is removed, and
the exception is re-raised. The SavedModel export, `load_model` and the `get_settings` signature are unchanged.

Callers call `export_tflite_model` explicitly: `ExperimentSummarizeService` after its Postgres writes,
`ModelRestorer` after `export_model`.

## Rejected alternatives
| Alternative | Why not |
|-------------|---------|
| Replace the SavedModel with TFLite | the validator and restorer read the SavedModel `get_settings` signature |
| Embed labels in TFLite metadata / string output ops | string ops need Select TF ops or are risky on a builtin-only runtime; sidecar is simpler |
| Select TF ops, int8 / float16 quantization | device runtime assumed builtin-only; goal 1 (accuracy) not yet measured against quantization |
| Call the export from inside `export_model` | a conversion failure would abort before the Postgres writes in the summarize service |

## Consequences
- Positive: every exported model ships as a deployable `.tflite`; reproducible via `main.py` / `export-best-model.py`.
- Accepted cost: a model that does not convert aborts the grid at that step (nothing in `Experiments.run` catches);
  the SavedModel and Postgres rows are already written, but `experiment.finish()` is skipped.
- Follow-ups: `tf.lite.Interpreter` is deprecated in TF 2.20 in favour of LiteRT (`ai_edge_litert`); moving to it
  needs a dependency approval (rule C2) and its own ticket.

## Revisit when
The device runtime needs Select TF ops or quantization, or the exporter moves to LiteRT.
