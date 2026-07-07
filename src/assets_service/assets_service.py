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

    def _det_data_set_label(self, labels: str):
        labels = labels.split(',')
        labels.sort()
        return ','.join(labels)

    def _get_data_set_fingerprint(self):
        details = self.experiment_model_service.get_data_set_details(experiment_id=self.experiment_id)
        finger_print = f'{self._det_data_set_label(details.labels).replace(",", "_")}-{details.duration}'
        return finger_print

    def _get_model_fingerprint(self):
        details = self.experiment_model_service.get_data_set_details(experiment_id=self.experiment_id)
        finger_print = f'{self._det_data_set_label(details.labels).replace(",", "_")}-{details.duration}-{details.af_type.value}'
        return finger_print

    def get_assets_path(self):
        return ASSETS_PATH

    def get_data_set_path(self):
        finger_print = self._get_data_set_fingerprint()
        return join(ASSETS_PATH, f'{self.out_data_set_name}_{finger_print}')

    def get_experiment_path(self):
        return self.experiment_path

    def get_validation_records_path(self):
        return join(ASSETS_PATH, self.validation_records_folder_name)

    def get_model_path(self):
        return self.model_path

    def get_models_path(self, model_name: str = None):
        if model_name is None:
            return join(ASSETS_PATH, 'models'), f'model-{self._get_model_fingerprint()}'
        return join(ASSETS_PATH, 'models'), model_name