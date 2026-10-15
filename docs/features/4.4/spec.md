# 4.4 — Export the model in TFLite format

## Problem
`ModelExporter.export_model` (`src/model_exporter/model_exporter.py:22`) writes only a TF SavedModel. The model is
deployed to a device (see `docs/BUSINESS_MODEL.md`), and a SavedModel is not a runnable on-device artefact: the
device needs a `.tflite` flatbuffer. Today the conversion is a manual step outside the repo.

## Goal / business value
Ranked goal 3 (small, fast, exportable models): every exported model also ships as a `.tflite` file next to the
SavedModel. Goal 1 (accuracy) must not drop: the default conversion is float32, no quantization.

## Scope
- in: `ModelExporter` gains a public `export_tflite_model`; `export_model` is unchanged. The two existing callers
  (`ExperimentSummarizeService`, after its DB writes, and `ModelRestorer`) call it explicitly.
- in: label names, `af_type` and `duration` (what `get_settings` carries in the SavedModel) are written next to the
  `.tflite` so the device can interpret the output.
- **out (explicit):**
  - removing or changing the SavedModel export, `load_model`, or the `get_settings` signature
    (`model_result_parser.py:63` and the validator depend on them)
  - quantization (int8 / float16) and a representative dataset
  - validating the `.tflite` model on `assets/validation-records` through the Interpreter
  - `train-rare-model.py`, which does not call `export_model` today
  - the device-side loader (not in this repo)
  - fixing the `input_shape` NameError in `train-rare-model.py:63`

## Acceptance criteria
- [x] AC1 Given a trained Keras model, when `export_tflite_model` runs for it, then `<export path>/model.tflite` exists beside the SavedModel.
- [ ] AC6 Given `ExperimentSummarizeService.summarize`, then the TFLite export runs after the Postgres result/plot writes, so a conversion failure cannot lose them. (call order read in the diff and the module compiles and imports; the pipeline was not run, see Q5)
- [x] AC2 Given the `.tflite` file and a random float32 input of the model's input shape, when run through `tf.lite.Interpreter`, then its output matches the Keras model's `predict` within 1e-4.
- [x] AC3 Given the export, then `model_settings.json` next to it holds `af_type` and `duration` equal to the reloaded SavedModel's `get_settings` signature, and `labels` equal to the `label_names` the SavedModel's `__call__` returns; it also holds `input_shape` equal to `model.input_shape[1:]`.
- [x] AC4 The SavedModel export, `load_model` and `get_settings` behave exactly as before.
- [x] AC5 A TFLite conversion failure is logged through `Logger('ModelExporter')` and re-raised (rule D5), and leaves no `model.tflite` or `model_settings.json` in that folder, including stale ones from a previous export (files are written under a temporary name and renamed only when both are ready).

## Edge cases
| Case | Expected behaviour | Decided by |
|------|--------------------|-----------|
| Model has a recurrent layer (GRU/LSTM) | converted with a fixed batch of 1 so builtin ops suffice; the `.tflite` input is `[1] + input_shape` | found when `export-best-model.py` failed on a GRU model |
| Model contains an op with no TFLite builtin | conversion raises with the converter's message; no silent fallback to Select TF ops | assumption (device runtime unknown) |
| Export path already holds a `.tflite` | overwritten on success, same as the SavedModel is; removed on failure (AC5) | AC5 |
| `labels` is a numpy array vs list | stored as a list of str in JSON | code |

## Open questions
| # | Question | Decision (maintainer, 2026-10-15) | Needs a human? |
|---|----------|-----------------------------------|----------------|
| 1 | Does TFLite replace the SavedModel or sit beside it? | Beside it: the validator and restorer read the SavedModel signature | decided |
| 2 | Which device runtime loads the file, and does it support Select TF ops? | Builtin ops only, float32 | decided |
| 3 | Is a sidecar JSON acceptable for labels/settings, or must they be embedded in the flatbuffer (TFLite metadata)? | Sidecar JSON | decided |
| 4 | Where does a TFLite failure land? | Option A: the TFLite export is called by the callers after the DB writes, not from inside `export_model` | decided |
| 5 | Verify with the real `export-best-model.py`? | No: it would overwrite `assets/models/model-*`; reported under "Not verified" | decided |
