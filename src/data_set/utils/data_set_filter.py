import shutil
import tensorflow as tf
from src.data_set.utils.data_set_splitter import DataSetFileWorker
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser
from src.utils.logger.logger_service import Logger
from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import ASSETS_PATH, FRAGMENT_LENGTH
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class DataSetFilter(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str], assets_service: IAssetsService, af_strategy: IAFStrategy):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.assets_service = assets_service
    self.model_parser = ModelResultParser(af_strategy=af_strategy)
    self.logger = Logger('DataSetFilter')
    self.except_sets = []
    self.except_labels = []

  def _get_model(self, duration: float):
     model_path = self.files.join(self.assets_service.get_assets_path(), 'models', f'model_{duration}')
     if not self.files.is_exist(model_path):
       return None
     return tf.saved_model.load(model_path)

  def filter(self, duration: float):
    self.logger.log('Start filtering', color='blue')
    model = self._get_model(duration)
    if not model:
      self.logger.log('Model not found. Filtering skipped.', color='red')
      return
    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      if set_name in self.except_sets:
        continue
      if label in self.except_labels:
        continue
      if len(signal) >= FRAGMENT_LENGTH:
        signal_label, _ = self.model_parser.parse(model=model, x=signal)
        if label != signal_label:
          out_folder = self.files.join(self.out_path, set_name, signal_label)
          self.files.create_folder(out_folder)
          to_path = self.files.join(out_folder, file)
          from_path = self.files.join(path, file)
          self.logger.log(f'moving {from_path} to {to_path}', color='yellow')
          shutil.move(from_path, to_path)

    self.logger.log('End filtering', color='blue')
