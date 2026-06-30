from abc import ABC, abstractmethod

import numpy as np

from src.utils.audio_features.types import AFTypes


class IAFStrategy(ABC):
  @abstractmethod
  def get_audio_feature(self, signal: np.ndarray):
    pass

  @abstractmethod
  def save_audio_feature(self, af: np.ndarray, label: str):
    pass

  @property
  def AFType(self) -> AFTypes:
    pass
