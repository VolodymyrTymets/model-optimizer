from src.utils.audio_features.types import AFTypes

ASSETS_PATH = 'assets'

EMULATE_MODE = False
sr = 44000
af_type = AFTypes.mfcc
DURATION = 0.5
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
n_mels = 64
n_mfcc = 64
labels = ['noise', 'stimulation', 'breath']
labels_colors = ['blue', 'red', 'green']
sub_sets = ['train', 'test']

