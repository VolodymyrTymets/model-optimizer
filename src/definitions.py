import os as _os
from pathlib import Path as _Path
from typing import Dict as _Dict, List as _List

from dotenv import load_dotenv as _load_dotenv

from src import definitions_emg as _emg
from src import definitions_sound as _sound

# Anchored to this file so the launch directory does not matter; a variable already set in the shell wins.
_load_dotenv(_Path(__file__).resolve().parent.parent / '.env', override=False)

_TASK_ENV_VAR: str = 'MODEL_OPTIMIZER_TASK'
_PROFILES = {'sound': _sound, 'emg': _emg}
# Unset means sound, which is what the pipeline did before profiles existed.
TASK: str = _os.environ.get(_TASK_ENV_VAR, 'sound')
if TASK not in _PROFILES:
    # No fallback: a typo here would silently train on the wrong data set.
    raise ValueError(f"{_TASK_ENV_VAR}={TASK!r} is not valid; use one of: {', '.join(sorted(_PROFILES))}")
_profile = _PROFILES[TASK]

# Task-specific values, see src/definitions_<task>.py
DEFAULT_DATABASE_URL: str = _profile.DEFAULT_DATABASE_URL
DATA_SET_NAME: str = _profile.DATA_SET_NAME
VALIDATION_RECORDS_COUNT: int = _profile.VALIDATION_RECORDS_COUNT
sr: int = _profile.sr
DURATION: float = _profile.DURATION
labels: _List[str] = _profile.labels
labels_colors: _Dict[str, str] = _profile.labels_colors

# Shared by every task
ASSETS_PATH = 'assets'
VERBOSE = True
EMULATE_MODE = False
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
n_mels = 64
n_mfcc = 64
sub_sets = ['train', 'test']
keet_prefix = 'keep'
# Share of the fragments in every validation record that must be keep-flagged, so record_acc rests on trusted labels.
KEEP_RECORDS_SHARE: float = 0.3
