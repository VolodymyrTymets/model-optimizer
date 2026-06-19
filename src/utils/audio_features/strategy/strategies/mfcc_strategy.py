import numpy as np
from src.audio_features.types import AFTypes
from src.definitions import n_mfcc
from src.audio_features.strategy.strategies.base_strategy import BaseStrategy


class MFCCStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.af_type = AFTypes.mfcc

  def get_audio_feature(self, signal: np.ndarray):
    return self.features.mfcc(signal=signal, sr=self.sr, n_mfcc=n_mfcc)


