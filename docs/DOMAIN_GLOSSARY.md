# Domain glossary

> One meaning per term. Drives naming in code, database and logs. Meanings below were inferred from the code and are marked **to confirm** where the maintainer should check them.

| Term | Code identifier | Means exactly | Not to be confused with |
|------|-----------------|---------------|-------------------------|
| Task (profile) | `MODEL_OPTIMIZER_TASK`, `TASK`, `src/definitions_<task>.py` | The signal type a run works on, `sound` or `emg`; fixes `sr`, `DURATION`, `labels`, data-set name and database | a data set, or an experiment |
| Experiment | `Experiment`, `ExperimentModel` | One model-search run for a given data-set configuration and model settings; has steps and an end time | `Experiments` (the orchestrator that creates many) |
| Experiment step | `ExperimentStep`, `ExperimentStepModel` | One candidate model configuration trained inside an experiment, with its metrics | an epoch |
| Data set | `DataSetCooker`, `DataSetImporter`, `assets/data_set_*` | Cooked train/val/test fragments of `DURATION` seconds, labelled, with features applied, and augmentation applied only once it is filtered (ADR-008) | raw recordings |
| Fragment | `FRAGMENT_LENGTH` | `sr * DURATION` samples: the unit a model classifies | a whole record |
| Record | `validation-records`, `ModelRecordLabeler` | A full-length recording used for validation, scored per record (`record_acc`) | fragment |
| Label | `labels` | A class name, e.g. `noise`, `stimulation` | audio feature type |
| Audio feature (AF) | `AFTypes`, `IAFStrategy` | A signal representation fed to the model: `wave, ae, rms, zcr, fft, stft, ber, sc, bw, mel, mfcc` (**to confirm** abbreviations `ae`, `ber`, `sc`, `bw`) | augmentation |
| Argumentation (sic) | `ArgumentationTypes` | Data augmentation: `normalization, time_stretch, pitch_shift, time_shift, nothing`. The spelling is the real identifier; the concept is "augmentation" | argument |
| Fingerprint | `_get_data_set_fingerprint` | Sorted labels + duration (+ AF type for models) used to name data-set and model folders | a hash |
| TFLite model | `ModelExporter.export_tflite_model`, `model.tflite` | The float32 flatbuffer copied to the device; returns probabilities only | the SavedModel |
| Model settings sidecar | `model_settings.json` | `labels`, `af_type`, `duration`, `sample_rate`, `input_shape` for the TFLite model, mirroring `get_settings` | the `get_settings` signature |
| Filtering | `DataSetFilter`, `filter-data-set.py`, `__filtered__` | Moving a cooked fragment to the label folder a model predicts when that differs from its current folder; the `__filtered__` folder marks a data set as done | augmentation, validation |
| Keep flag | `keet_prefix` (`'keep'`) | A word in a file name (`keep` anywhere in the name) for a fragment that is certainly in the right label folder; filtering never moves it | |
| Model name | `AssetsService.get_models_path`, `assets/models/<name>` | `model-<labels joined by _>-<duration>-<af_type>`, the model fingerprint with a `model-` prefix; the argument of `filter-data-set.py` | the data-set folder name |
| Rare model | `train-rare-model.py`, `assets/rare-model-image` | A single hand-built model for a rare class / image data set, outside the grid search (**to confirm**) | the grid search |

## Naming rules that follow from the above
- Keep the existing spellings `Argumentation`, `mode_*` (for `model_*`) in identifiers — they are import paths; use "augmentation" / "model" in prose and new names (rule C1).
- New classifier classes are `LabelType`-style enums in `src/data_set/types.py` / `src/model_schema/model_schema_types.py`, not strings.

## Status/enum vocabularies
| Enum | Values | Meaning of each |
|------|--------|-----------------|
| `LayerType`, `ActivationType`, `OptimizerType`, `RegularizerType`, `LossType` | in `src/model_schema/model_schema_types.py` | Search space for the tuner |
| `ArgumentationTypes` | `normalization`, `time_stretch`, `pitch_shift`, `time_shift`, `nothing` | Augmentations applied when cooking a data set |
| `AFTypes` | see above | Feature extraction strategy |
