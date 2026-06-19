import librosa
import soundfile as sf
from abc import ABC, abstractmethod

import numpy as np

class IWavFile(ABC):
  @abstractmethod
  def read(self, file_path: str):
    pass

  def write(self, file_path: str, signal: np.ndarray, sr: int):
    pass


class WavFiles(IWavFile):
  def read(self, file_path: str):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

  def write(self, file_path: str, signal: np.ndarray, sr: int):
    sf.write(file_path, signal, samplerate=sr)