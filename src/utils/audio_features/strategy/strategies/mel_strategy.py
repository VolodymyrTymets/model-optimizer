import numpy as np

from src.definitions import n_mels
from src.audio_features.strategy.strategies.base_strategy import BaseStrategy
from src.audio_features.types import AFTypes


class MelStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.af_type = AFTypes.mel

  def get_audio_feature(self, signal: np.ndarray):
    return self.features.melspectogram(signal=signal, sr=self.sr, frame_length=self.frame_length,
                                       hop_length=self.hop_length, n_mels=n_mels)
