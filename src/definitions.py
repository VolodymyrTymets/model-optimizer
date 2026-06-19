ASSETS_PATH = 'assets'

# data set config
dataset_name = f'data_set'
sr = 44000
DURATION = 0.05
FRAGMENT_LENGTH = int(sr / (1 / DURATION))
frame_length = 512
hop_length = frame_length // 4
split_frequency = 2000
n_mels = 64
n_mfcc = 64
labels = ['noise', 'stimulation', 'breath']

EPOCHS = 100
labels_colors = ['blue', 'red', 'green']
sub_sets = ['train', 'test']

