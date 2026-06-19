from enum import Enum


class AFTypes(Enum):
  wave = 'wave'
  ae = 'ae'
  rms = 'rms'
  zcr = 'zcr'
  fft = 'fft'
  stft = 'stft'
  ber = 'ber'
  sc = 'sc'
  bw = 'bw'
  mel = 'mel'
  mfcc = 'mfcc'

class ArgumentationTypes(Enum):
  normalization = 'normalization'
  time_stretch = 'time_stretch'
  pitch_shift = 'pitch_shift'
  time_shift = 'time_shift'