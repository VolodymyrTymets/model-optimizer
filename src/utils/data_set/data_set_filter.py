import numpy as np
import shutil
import tensorflow as tf
from src.data_set.data_set_file_worker import DataSetFileWorker
from src.audio_features.types import AFTypes
from src.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.logger.logger_service import Logger
from src.audio_features.signal_transformer import SignalTransformer
from src.neural_network.utils import label_by_model

from src.definitions import ASSETS_PATH, FRAGMENT_LENGTH, sr as SR, frame_length, hop_length


class DataSetFilter(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)
    self.logger = Logger('DataSetFilter')

  def filter(self, except_sets: list['str'], except_labels: list[str], model_name: str, strategy_type: AFTypes):
    self.logger.log('Start filtering')
    model = tf.saved_model.load(self.files.join(self.files.ASSETS_PATH, 'models', model_name))
    strategy = AFStrategyFactory(sr=SR, frame_length=frame_length, hop_length=hop_length).create_strategy(strategy_type)
    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      if set_name in except_sets:
        continue
      if label in except_labels:
        continue
      if len(signal) >= FRAGMENT_LENGTH:
        af = strategy.get_audio_feature(signal=signal)
        signal_label, = label_by_model(model, af)
        if label != signal_label:
          out_folder = self.files.join(ASSETS_PATH, self.in_path, set_name, '__filtered__', signal_label)
          self.files.create_folder(out_folder)
          to_path = self.files.join(out_folder, file)
          from_path = self.files.join(path, file)
          self.logger.log('{} -> {}'.format(from_path, to_path), color='red')
          shutil.move(from_path, to_path)

    self.logger.log('End filtering')
