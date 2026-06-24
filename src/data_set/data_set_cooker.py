from os.path import join, exists
import shutil

from src.data_set.utils.data_set_splitter import DataSetSplitter
from src.data_set.utils.data_set_record_generator import DataSetRecordGenerator
from src.data_set.utils.data_set_transformer import DataSetTransformer
from src.definitions import labels, sub_sets
from src.data_set.types import ArgumentationTypes
from src.utils.logger.logger_service import Logger
from src.definitions import ASSETS_PATH


class DataSetCooker:
    def __init__(self, experiment_id: int):
        self.experiment_path = join(ASSETS_PATH, f'experiment-{experiment_id}')
        self.in_data_set_name = 'rare_data_set'
        self.out_data_set_name = 'data_set'
        self.validation_records_folder_name = 'records'
        self.model_path = join(self.experiment_path, 'model')
        self.data_set_path = join(self.experiment_path, self.out_data_set_name)
        self.data_set_splitter = DataSetSplitter(in_path=join(self.experiment_path, self.in_data_set_name),
                                                 out_path=self.data_set_path, sub_sets=sub_sets, labels=labels)
        self.data_set_transformer = DataSetTransformer(in_path=self.data_set_path, out_path=self.data_set_path,
                                                       sub_sets=sub_sets,
                                                       labels=labels)
        self.data_set_record_generator = DataSetRecordGenerator(in_path=self.data_set_path,
                                                                out_path=join(self.experiment_path,
                                                                              self.out_data_set_name),
                                                                sub_sets=sub_sets, labels=labels)

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
        if exists(self.data_set_path):
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
        if exists(join(self.experiment_path, self.out_data_set_name, self.validation_records_folder_name)):
            self.logger.log(f'Train records already generated', color='green')
            return
        for _ in range(record_count):
            self.data_set_record_generator.generate_test_record(duration=duration, except_sets=['train'],
                                                                except_labels=[])

    def prepare(self, duration: float = 0.5, argumentation_types=list[ArgumentationTypes]):
        self._copy_data_set()
        self._split_data_set(duration)
        self._argument_data_set(argumentation_types=argumentation_types)
        self._generate_records(duration, record_count=10)

    def get_data_set_path(self):
        return self.data_set_path

    def get_experiment_path(self):
        return self.experiment_path

    def get_validation_records_path(self):
        return join(self.data_set_path, self.validation_records_folder_name)

    def get_model_path(self):
        return self.model_path
