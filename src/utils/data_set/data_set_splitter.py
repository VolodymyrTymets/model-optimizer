from src.data_set.data_set_file_worker import DataSetFileWorker
from src.logger.logger_service import Logger
from src.audio_features.signal_transformer import SignalTransformer

from src.definitions import ASSETS_PATH, sr as SR, frame_length, hop_length


class DataSetSplitter(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.logger = Logger('DataSetSplitter')
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)

  def split(self, duration: float):
    self.logger.log('Start splitting')
    self.logger.log(f'Creating directory: {self.files.join(ASSETS_PATH, self.out_path)}')
    self.files.create_folder(self.files.join(ASSETS_PATH, self.out_path))
    for i, data in enumerate(self.read_data_set()):
      signal, sr, set_name, label, path, file = data
      fragments = self.transformer.split(signal=signal, duration=duration)
      for time_index, fragment in enumerate(fragments):
        self.write_signal(fragment, sr, path, str(i + time_index))
    self.logger.log('End splitting')
