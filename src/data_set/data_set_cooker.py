import shutil
from os.path import join, exists

from src.assets_service.assets_service import AssetsService
from src.data_set.utils.data_set_filter import DataSetFilter
from src.data_set.utils.data_set_splitter import DataSetSplitter
from src.data_set.utils.data_set_record_generator import DataSetRecordGenerator
from src.data_set.utils.data_set_transformer import DataSetTransformer
from src.definitions import labels, sub_sets, EMULATE_MODE
from src.data_set.types import ArgumentationTypes
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger
from src.definitions import ASSETS_PATH, VALIDATION_RECORDS_COUNT


class DataSetCooker:
    def __init__(self, experiment_id: int, af_strategy: IAFStrategy):
        self._asset_service = AssetsService(experiment_id=experiment_id)
        self.experiment_path = join(ASSETS_PATH, f'experiment-{experiment_id}')
        self.model_path = join(self.experiment_path, 'model')
        datasets_path = self._asset_service.get_data_set_path()
        self.data_set_splitter = DataSetSplitter(in_path=join(ASSETS_PATH, 'data_set'),
                                                 out_path=datasets_path, sub_sets=sub_sets, labels=labels)
        self.data_set_transformer = DataSetTransformer(in_path=datasets_path, out_path=datasets_path,
                                                       sub_sets=sub_sets,
                                                       labels=labels)
        self.data_set_record_generator = DataSetRecordGenerator(in_path=datasets_path,
                                                                out_path=self._asset_service.get_validation_records_path(),
                                                                sub_sets=sub_sets, labels=labels)
        self.data_set_filter = DataSetFilter(in_path=datasets_path, out_path=datasets_path,
                                             sub_sets=sub_sets, labels=labels, af_strategy=af_strategy,
                                             assets_service=self._asset_service, experiment_id=experiment_id)

        self.logger = Logger('DataSet')
        self._datasets_path = datasets_path

    def remove_data_set(self):
        if exists(self._datasets_path):
            shutil.rmtree(self._datasets_path)
            self.logger.log(f'Previous Data set removed', color='green')

    def _split_data_set(self, duration: float = 0.5):
        if exists(self._asset_service.get_data_set_path()):
            self.logger.log(f'Data set already splitted', color='green')
            return
        self.logger.log(f'Splitting data set into train and test sets with duration: {duration}', color='blue')
        self.data_set_splitter.split(duration)

    def _argument_data_set(self, argumentation_types: list[ArgumentationTypes]):
        if EMULATE_MODE:
            return False
        if self.data_set_filter.is_filtered() is False:
            self.logger.log(f'Data set is not filtered. Skipping argumentation', color='red')
            return False
        self.logger.log(
            f'Transforming data set with argumentation types: {",".join([x.value for x in argumentation_types])}',
            color='blue')
        return self.data_set_transformer.argument(argumentation_types=argumentation_types, except_sets=['test'],
                                           except_labels=['noise'])

    def _generate_records(self, duration: float = 0.5, record_count: int = 10, is_need_to_regenerate: bool = False):
        if EMULATE_MODE:
            return
        self.logger.log(f'Generating records for train and test sets', color='blue')
        if is_need_to_regenerate:
            shutil.rmtree(self._asset_service.get_validation_records_path())
        if exists(self._asset_service.get_validation_records_path()):
            self.logger.log(f'Validation records already generated', color='green')
            return
        for _ in range(record_count):
            self.data_set_record_generator.generate_test_record(duration=duration, except_sets=['train'],
                                                                except_labels=[])

    def _filter_data_set(self, duration: float = 0.5):
        if EMULATE_MODE:
            return False
        return self.data_set_filter.filter(duration=duration)

    def step_prepare(self, duration: float, argumentation_types: list[ArgumentationTypes]):
        is_filtered = self._filter_data_set(duration)
        print(f'is_filtered: {is_filtered}')
        if is_filtered is False:
            return False
        is_argumeted = self._argument_data_set(argumentation_types=argumentation_types)
        print(f'is_argumeted: {is_argumeted}')
        return is_filtered or is_argumeted

    def prepare(self, duration: float, argumentation_types: list[ArgumentationTypes]):
        self.remove_data_set()
        self._split_data_set(duration)
        is_filtered = self._filter_data_set(duration)
        self._argument_data_set(argumentation_types=argumentation_types)
        self._generate_records(duration, record_count=VALIDATION_RECORDS_COUNT, is_need_to_regenerate=is_filtered)
