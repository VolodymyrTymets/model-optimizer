import random
import glob
import numpy as np

from src.data_set.types import ArgumentationTypes
from src.data_set.utils.data_set_file_worker import DataSetFileWorker
from src.utils.audio_features.signal_transformer import SignalTransformer

from src.definitions import sr as SR, frame_length, hop_length


class DataSetTransformer(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)

  def _is_argumentation_done(self, path: str, argumentation_type: ArgumentationTypes):
    path = self.files.join(path, f'{argumentation_type.value}*')
    self.logger.log(f'Search for: {path}')
    files = glob.glob(path)
    if files:
      return True
    return False

  def _pitch_shift(self, signal: np.ndarray, sr: int, path: str):
    shifted = self.transformer.pitch_shift(signal, sr)
    self.write_signal(shifted, sr, path, f'{ArgumentationTypes.pitch_shift.value}')

  def _time_shift(self, signal: np.ndarray, sr: int, path: str):
    shifted = self.transformer.time_shift(signal, 1)
    self.write_signal(shifted, sr, path, f'{ArgumentationTypes.time_shift.value}')

  def _time_stretch(self, signal: np.ndarray, sr: int, path: str):
    random_rate = random.uniform(0.5, 1)
    stretched = self.transformer.time_stretch(signal, random_rate)[:len(signal)]
    self.write_signal(stretched, sr, path, f'{ArgumentationTypes.time_stretch.value}')

  def _normalize(self, signal: np.ndarray, sr: int, path: str):
    normalize_fragments = self.transformer.normalize(signal)
    for normalize_index, n_fragment in enumerate(normalize_fragments):
      self.write_signal(n_fragment, sr, path, f'{ArgumentationTypes.normalization.value}_{normalize_index}')

  def argument(self, argumentation_types=list[ArgumentationTypes], except_sets: list['str'] = [],
               except_labels: list[str] = []):

    normalization_done_for = []
    for set_name in self.set_names:
      if set_name in except_sets:
        continue
      for label in self.labels:
        if label in except_labels:
          continue
        for argumentation_type in argumentation_types:
          is_done = self._is_argumentation_done(path=self.files.join(self.out_path, set_name, label), argumentation_type=argumentation_type)
          if is_done:
            self.logger.log(f'Argumentation {argumentation_type.value} is done', color='green')
            normalization_done_for.append(argumentation_type)


    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      # transformations are only for train set
      if set_name in except_sets:
        continue
      # transformations are only for train set
      if label in except_labels:
        continue

      if ArgumentationTypes.normalization in argumentation_types:
        if ArgumentationTypes.normalization in normalization_done_for:
          continue
        self._normalize(signal, sr, path)
      if ArgumentationTypes.pitch_shift in argumentation_types:
        if ArgumentationTypes.pitch_shift in normalization_done_for:
          continue
        self._pitch_shift(signal, sr, path)
      if ArgumentationTypes.time_shift in argumentation_types:
        if ArgumentationTypes.time_shift in normalization_done_for:
          continue
        self._time_shift(signal, sr, path)
      if ArgumentationTypes.time_stretch in argumentation_types:
        if ArgumentationTypes.time_stretch in normalization_done_for:
          continue
        self._time_stretch(signal, sr, path)
