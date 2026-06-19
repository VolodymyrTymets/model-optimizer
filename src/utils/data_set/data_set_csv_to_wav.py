import pandas as pd
import soundfile as sf

from src.data_set.data_set_file_worker import DataSetFileWorker
from src.logger.logger_service import Logger
from src.audio_features.signal_transformer import SignalTransformer

from src.definitions import sr as SR, frame_length, hop_length


class DataSetCSVToWav(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)
    self.logger = Logger('AugmentationPipline')

  def read_data_set(self, log=True):
    for set_name in self.set_names:
      for label in self.labels:
        path = self.files.join(self.files.ASSETS_PATH, self.in_path, set_name, label)
        files = self.files.get_only_files(path)

        for file in files:
          yield path, file

  def csv_to_wav(self):
    self.logger.log('Start converting csv to wav')
    for path, file in self.read_data_set(log=False):

      # assume we have columns 'time' and 'value'
      self.logger.log(f'--'
                      f'> read from: {self.files.join(path, file)}')
      df = pd.read_csv(self.files.join(path, file))

      times = []

      for i in range(len(df)):
        times.append(int(df.iloc[i][0]))

      # compute sample rate, assuming times are in seconds
      times = [x / 1000 for x in times]
      n_measurements = len(times)
      timespan_seconds = times[-1] - times[0]
      sample_rate_hz = int(n_measurements / timespan_seconds)

      headers = df.columns
      for h_index in range(len(headers)):
        if h_index == 0:
          continue
        name = headers[h_index]

        data = df[name].values
        # convert to int16
        data = [(x / 32767) * -1 if i % 2 == 0 else x / 32767 for i, x in enumerate(data)]
        # write data
        out_path = self.files.join(path, name)
        self.files.create_folder(out_path)
        sf.write(self.files.join(out_path, file.replace('.csv', '.wav')), data, sample_rate_hz)

    self.logger.log('End converting csv to wav')
