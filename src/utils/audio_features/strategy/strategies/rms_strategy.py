import numpy as np
import matplotlib.pyplot as plt
from src.audio_features.strategy.strategies.base_strategy import BaseStrategy
from src.audio_features.audio_features import TimeDomainFeatures
from src.audio_features.types import AFTypes


class RMStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.features = TimeDomainFeatures()
    self.af_type = AFTypes.rms
    fig, ax = plt.subplots()
    self.fig = fig
    self.ax = ax

  def get_audio_feature(self, signal: np.ndarray):
    rms = self.features.RSME(signal=signal, frame_length=self.frame_length, hop_length=self.hop_length)
    return rms
