import random

import numpy as np

from src.audio_features.types import ArgumentationTypes
from src.data_set.data_set_file_worker import DataSetFileWorker
from src.logger.logger_service import Logger
from src.audio_features.signal_transformer import SignalTransformer

from src.definitions import sr as SR, frame_length, hop_length


class DataSetTransformer(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)
    self.logger = Logger('AugmentationPipline')

  def _pitch_shift(self, signal: np.ndarray, sr: int, path: str):
    shifted = self.transformer.pitch_shift(signal, sr)
    self.write_signal(shifted, sr, path, f'p_shift')

  def _time_shift(self, signal: np.ndarray, sr: int, path: str):
    shifted = self.transformer.time_shift(signal, 1)
    self.write_signal(shifted, sr, path, f't_shift')

  def _time_stretch(self, signal: np.ndarray, sr: int, path: str):
    random_rate = random.uniform(0.5, 1)
    stretched = self.transformer.time_stretch(signal, random_rate)[:len(signal)]
    self.write_signal(stretched, sr, path, f't_stretch')

  def _normalize(self, signal: np.ndarray, sr: int, path: str):
    normalize_fragments = self.transformer.normalize(signal)
    for normalize_index, n_fragment in enumerate(normalize_fragments):
      self.write_signal(n_fragment, sr, path, f'norm_{normalize_index}')

  def argument(self, argumentation_types=list[ArgumentationTypes], except_sets: list['str'] = [],
               except_labels: list[str] = []):
    self.logger.log('Start argumentation')
    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      # transformations are only for train set
      if set_name in except_sets:
        continue
      # transformations are only for train set
      if label in except_labels:
        continue
      if ArgumentationTypes.normalization.value in argumentation_types:
        self._normalize(signal, sr, path)
      if ArgumentationTypes.pitch_shift.value in argumentation_types:
        self._pitch_shift(signal, sr, path)
      if ArgumentationTypes.time_shift.value in argumentation_types:
        self._time_shift(signal, sr, path)
      if ArgumentationTypes.time_stretch.value in argumentation_types:
        self._time_stretch(signal, sr, path)
    self.logger.log('End argumentation')
