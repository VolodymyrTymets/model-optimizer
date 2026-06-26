from tensorboard.compat import tf

from src.assets_service.assets_service_interface import IAssetsService
from src.model_exporter.model_weights_exporter.model_weights_exporter_interface import IModelWeightsExporter
from src.utils.files import Files
from src.utils.logger.logger_service import Logger


class ModelWeightsExporter(IModelWeightsExporter):
    def __init__(self, asset_service: IAssetsService):
        self.files = Files()
        self.loger = Logger('ModelWeightsExporter')
        self.target_path = self.files.join(asset_service.get_experiment_path(), 'weights')
        self.files.create_folder(self.target_path)

    def _get_weights_path(self, step: int):
        return self.files.join(self.target_path, f'iw_{step}.weights.h5')

    def export_weights(self, model, step: int):
        model.save_weights(self._get_weights_path(step))

    def import_weights(self, model, step: int) -> tf.keras.Model:
        model.load_weights(self._get_weights_path(step))
        return model
