# Default local store for experiment-progress logging.
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/model-optimizer-emg-1"

ASSETS_PATH = 'assets'
DATA_SET_NAME = 'data_set_emg'
VERBOSE = True
EMULATE_MODE = False
SKIP_FILTER = True
VALIDATION_RECORDS_COUNT = 10
sr = 44100
DURATION = 1
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
n_mels = 64
n_mfcc = 64
# RLN
# labels = ['noise', 'stimulation', 'breath']
# labels_colors = {
#     'noise': 'blue',
#     'stimulation': 'red',
#     'breath': 'green'
# }
#EMG
labels = ['noise', 'WristExtension', 'WristFlexion'] #'Supination', 'Rest', 'Pronation', 'HandOpen', 'HandClose']
labels_colors = {
    'noise': 'blue',
    'WristExtension': 'red',
    'WristFlexion': 'green',
    # 'Supination': 'yellow',
    # 'Rest': 'black',
    # 'Pronation': 'purple',
    # 'HandOpen': 'orange',
    # 'HandClose': 'pink'
}
#speech
# labels = ['noise', 'down', 'up', 'go', 'left', 'right', 'stop', 'yes', 'no']
# labels_colors = {
#     'noise': 'blue',
#     'down': 'red',
#     'up': 'green',
#     'left': 'yellow',
#     'right': 'black',
#     'go': 'purple',
#     'stop': 'orange',
#     'yes': 'pink',
#     'no': 'brown'
# }
sub_sets = ['train', 'test']
keet_prefix = 'keep'

