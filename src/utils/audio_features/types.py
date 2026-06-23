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