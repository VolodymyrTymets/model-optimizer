import pickle

from tensorboard.compat import tf

from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import EMULATE_MODE
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_exporter.model_weights_exporter.model_weights_exporter_interface import IModelWeightsExporter
from src.utils.files import Files
from src.utils.logger.logger_service import Logger


class ModelWeightsExporter(IModelWeightsExporter):
    def __init__(self, asset_service: IAssetsService):
        self.files = Files()
        self.loger = Logger('ModelWeightsExporter')
        self.target_path = self.files.join(asset_service.get_experiment_path(), 'weights')
        self.files.create_folder(self.target_path)
        self.experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))

    def export_weights(self, model, step: int):
        if EMULATE_MODE:
            return
        weights = model.get_weights()
        serialized_weights = pickle.dumps(weights)
        self.experiment_step_model_service.save_final_weights(step, serialized_weights)

    def import_weights(self, model, step: int) -> tf.keras.Model:
        if EMULATE_MODE:
            return model
        weight = self.experiment_step_model_service.get_weights(step)
        if not weight:
            return model
        self.loger.log(f'Importing weights from {weight.id}', color='blue')
        restored_weights = pickle.loads(weight.data)
        model.set_weights(restored_weights)
        return model
