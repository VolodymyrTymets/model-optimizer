import numpy as np
import matplotlib.pyplot as plt
from src.audio_features.strategy.strategies.base_strategy import BaseStrategy
from src.definitions import split_frequency
from src.audio_features.types import AFTypes


class BERtrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.af_type = AFTypes.ber
    fig, ax = plt.subplots()
    self.fig = fig
    self.ax = ax

  def get_audio_feature(self, signal: np.ndarray):
    af = self.features.BER(signal=signal, sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length, split_frequency=split_frequency)
    return af
