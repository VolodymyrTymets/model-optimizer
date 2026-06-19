from abc import ABC, abstractmethod

import numpy as np

class IAFStrategy(ABC):
  @abstractmethod
  def get_audio_feature(self, signal: np.ndarray):
    pass

  @abstractmethod
  def save_audio_feature(self, af: np.ndarray, label: str):
    pass
