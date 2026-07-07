from os.path import join

from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import ASSETS_PATH
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.utils.logger.logger_service import Logger


class AssetsService(IAssetsService):
    def __init__(self, experiment_id):
        self.experiment_path = join(ASSETS_PATH, f'experiment-{experiment_id}')
        self.experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))
        self.out_data_set_name = 'data_set'
        self.validation_records_folder_name = 'validation-records'
        self.model_path = join(self.experiment_path, 'model')
        self.experiment_id = experiment_id


    def get_assets_path(self):
        return ASSETS_PATH

    def get_data_set_path(self):
        details = self.experiment_model_service.get_data_set_details(experiment_id=self.experiment_id)
        finger_print = f'{details.labels.replace(",", "_")}-{details.duration}'
        return join(ASSETS_PATH, f'{self.out_data_set_name}_{finger_print}')

    def get_experiment_path(self):
        return self.experiment_path

    def get_validation_records_path(self):
        return join(ASSETS_PATH, self.validation_records_folder_name)

    def get_model_path(self):
        return self.model_path