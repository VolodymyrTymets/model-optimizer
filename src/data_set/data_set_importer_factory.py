from src.data_set.data_set_importer import DataSetImporter
from src.data_set.image_data_set_importer import ImageDataSetImporter
from src.definitions import DATA_SET_TYPE
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy


def create_data_set_importer(experiment_id: int, af_strategy: IAFStrategy, duration: float):
  if DATA_SET_TYPE == 'image':
    return ImageDataSetImporter()
  return DataSetImporter(experiment_id=experiment_id, duration=duration, af_strategy=af_strategy)
