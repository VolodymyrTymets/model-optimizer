# Default local store for experiment-progress logging.
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/model-optimizer-image"

ASSETS_PATH = 'assets'
DATA_SET_NAME = 'brain-tumor-mri-dataset'
# 'audio' (wav + audio features) or 'image' (jpg, no audio features/record accuracy)
DATA_SET_TYPE = 'image'
IMAGE_SIZE = (256, 256)
IMAGE_COLOR_MODE = 'grayscale'
VERBOSE = True
EMULATE_MODE = False
SKIP_FILTER = True
VALIDATION_RECORDS_COUNT = 10
sr = 44100
DURATION = 0.5
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
n_mels = 64
n_mfcc = 64
# Brain tumor MRI (image)
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
labels_colors = {
    'glioma': 'red',
    'meningioma': 'green',
    'notumor': 'blue',
    'pituitary': 'orange'
}
# RLN
# labels = ['noise', 'stimulation', 'breath']
# labels_colors = {
#     'noise': 'blue',
#     'stimulation': 'red',
#     'breath': 'green'
# }
#EMG
# labels = ['noise', 'WristExtension', 'WristFlexion'] #'Supination', 'Rest', 'Pronation', 'HandOpen', 'HandClose']
# labels_colors = {
#     'noise': 'blue',
#     'WristExtension': 'red',
#     'WristFlexion': 'green',
#     # 'Supination': 'yellow',
#     # 'Rest': 'black',
#     # 'Pronation': 'purple',
#     # 'HandOpen': 'orange',
#     # 'HandClose': 'pink'
# }
#Speech
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

