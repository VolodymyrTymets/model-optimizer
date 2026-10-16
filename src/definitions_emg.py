from typing import Dict, List

# Per-task values for MODEL_OPTIMIZER_TASK=emg; shared constants live in src/definitions.py.
DEFAULT_DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/model-optimizer-emg"
DATA_SET_NAME: str = 'data_set_emg'
VALIDATION_RECORDS_COUNT: int = 10
sr: int = 8820
DURATION: float = 0.05
labels: List[str] = ['noise', 'stimulation']
labels_colors: Dict[str, str] = {
    'noise': 'blue',
    'stimulation': 'red',
}
