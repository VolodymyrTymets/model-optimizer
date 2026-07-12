# Default local store for experiment-progress logging.
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/model-optimizer"

ASSETS_PATH = 'assets'
VERBOSE = True
EMULATE_MODE = False
VALIDATION_RECORDS_COUNT = 10
sr = 44100
DURATION = 0.5
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
n_mels = 64
n_mfcc = 64
labels = ['noise', 'stimulation', 'breath']
labels_colors = {
    'noise': 'blue',
    'stimulation': 'red',
    'breath': 'green'
}
sub_sets = ['train', 'test']
keet_prefix = 'keep'

