from os.path import join

from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import ASSETS_PATH


class AssetsService(IAssetsService):
    def __init__(self, experiment_id):
        self.experiment_path = join(ASSETS_PATH, f'experiment-{experiment_id}')

        self.out_data_set_name = 'data_set'
        self.validation_records_folder_name = 'records'
        self.model_path = join(self.experiment_path, 'model')
        self.data_set_path = join(self.experiment_path, self.out_data_set_name)

    def get_data_set_path(self):
        return self.data_set_path

    def get_experiment_path(self):
        return self.experiment_path

    def get_validation_records_path(self):
        return join(self.data_set_path, self.validation_records_folder_name)

    def get_model_path(self):
        return self.model_path