import uuid

import numpy as np
from src.files import Files
from src.wav_files import WavFiles
from src.logger.logger_service import Logger
from src.definitions import ASSETS_PATH

class DataSetFileWorker:
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    self.in_path = in_path
    self.out_path = out_path
    self.labels = labels
    self.set_names = sub_sets
    self.files = Files()
    self.wave_file = WavFiles()
    self.logger = Logger('DataSetFileWorker')

  def get_in_path(self):
    return self.files.join(ASSETS_PATH, self.in_path)

  def write_signal(self, signal: np.ndarray, sr: int, file_path: str, prefix: str):
    out_path = file_path.replace(self.in_path, self.out_path)
    self.files.create_folder(out_path)
    file_name = self.files.join(out_path, f'{prefix}_' + uuid.uuid4().hex + '.wav')
    self.logger.log(f'--> write to: {file_name}')
    self.wave_file.write(file_name, signal, sr)
    return file_name

  def read_data_set(self, log: bool = True):
    ds_path = self.files.join(ASSETS_PATH, self.in_path)
    print('Start reading data set:' , ds_path)
    for set_name in self.set_names:
      for label in self.labels:
        try:
          path = self.files.join(ds_path, set_name, str(label))
        except Exception as e:
          print('Error in label:', [ds_path, set_name, label])
          print(e)
          continue
        files = self.files.get_only_files(path)

        for file in files:
          file_path = self.files.join(path, file)
          if log:
            self.logger.log(f'--> read from: {file_path}', 'blue')
          signal, sr = self.wave_file.read(file_path=file_path)

          yield signal, sr, set_name, label, path, file