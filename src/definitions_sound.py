from typing import Dict, List

# Per-task values for MODEL_OPTIMIZER_TASK=sound; shared constants live in src/definitions.py.
DEFAULT_DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/model-optimizer-sound"
DATA_SET_NAME: str = 'data_set_sound'
VALIDATION_RECORDS_COUNT: int = 40
sr: int = 44100
DURATION: float = 0.5
labels: List[str] = ['noise', 'stimulation', 'breath']
labels_colors: Dict[str, str] = {
    'noise': 'blue',
    'stimulation': 'red',
    'breath': 'green'
}
