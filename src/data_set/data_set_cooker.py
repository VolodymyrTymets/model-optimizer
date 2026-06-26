from os.path import join, exists
import shutil

from src.assets_service.assets_service import AssetsService
from src.data_set.utils.data_set_filter import DataSetFilter
from src.data_set.utils.data_set_splitter import DataSetSplitter
from src.data_set.utils.data_set_record_generator import DataSetRecordGenerator
from src.data_set.utils.data_set_transformer import DataSetTransformer
from src.definitions import labels, sub_sets
from src.data_set.types import ArgumentationTypes
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger
from src.definitions import ASSETS_PATH


class DataSetCooker:
    def __init__(self, experiment_id: int, af_strategy: IAFStrategy):
        self._asset_service = AssetsService(experiment_id=experiment_id)
        self.experiment_path = join(ASSETS_PATH, f'experiment-{experiment_id}')
        self.model_path = join(self.experiment_path, 'model')
        self.in_data_set_name = 'rare_data_set'
        datasets_path = self._asset_service.get_data_set_path()
        self.data_set_splitter = DataSetSplitter(in_path=join(self.experiment_path, self.in_data_set_name),
                                                 out_path=datasets_path, sub_sets=sub_sets, labels=labels)
        self.data_set_transformer = DataSetTransformer(in_path=datasets_path, out_path=datasets_path,
                                                       sub_sets=sub_sets,
                                                       labels=labels)
        self.data_set_record_generator = DataSetRecordGenerator(in_path=datasets_path,
                                                                out_path=datasets_path,
                                                                sub_sets=sub_sets, labels=labels)
        self.data_set_filter = DataSetFilter(in_path=datasets_path, out_path=datasets_path,
                                             sub_sets=sub_sets, labels=labels, af_strategy=af_strategy,
                                             assets_service=self._asset_service)

        self.logger = Logger('DataSet')

    def _copy_data_set(self):
        source_dir = join(ASSETS_PATH, 'data_set')
        destination_dir = join(self.experiment_path, self.in_data_set_name)
        if not exists(source_dir):
            raise Exception('Data set not found')
        if exists(destination_dir):
            self.logger.log(f'Data set already exists in {destination_dir}', color='green')
            return
        self.logger.log(f'Copy data set from {source_dir} to {destination_dir}')
        shutil.copytree(source_dir, destination_dir)
        self.logger.log(f'Data set copied to {destination_dir}', color='green')

    def _split_data_set(self, duration: float = 0.5):
        if exists(self._asset_service.get_data_set_path()):
            self.logger.log(f'Data set already splitted', color='green')
            return
        self.logger.log(f'Splitting data set into train and test sets with duration: {duration}', color='blue')
        self.data_set_splitter.split(duration)

    def _argument_data_set(self, argumentation_types=list[ArgumentationTypes]):
        self.logger.log(
            f'Transforming data set with argumentation types: {",".join([x.value for x in argumentation_types])}',
            color='blue')
        self.data_set_transformer.argument(argumentation_types=argumentation_types, except_sets=['test'],
                                           except_labels=[])

    def _generate_records(self, duration: float = 0.5, record_count: int = 10):
        self.logger.log(f'Generating records for train and test sets', color='blue')
        if exists(self._asset_service.get_validation_records_path()):
            self.logger.log(f'Train records already generated', color='green')
            return
        for _ in range(record_count):
            self.data_set_record_generator.generate_test_record(duration=duration, except_sets=['train'],
                                                                except_labels=[])

    def _filter_data_set(self, duration: float = 0.5):
        self.data_set_filter.filter(duration=duration)

    def prepare(self, duration: float = 0.5, argumentation_types=list[ArgumentationTypes]):
        self._copy_data_set()
        self._split_data_set(duration)
        self._filter_data_set(duration)
        self._argument_data_set(argumentation_types=argumentation_types)
        self._generate_records(duration, record_count=10)
