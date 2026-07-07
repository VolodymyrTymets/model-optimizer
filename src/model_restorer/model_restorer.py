from typing import Optional

from src.definitions import DURATION

from src.assets_service.assets_service import AssetsService
from src.data_set.data_set_importer import DataSetImporter
from src.experiment.experiment_step.experiment_step import ExperimentStep
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_exporter.model_exporter import ModelExporter
from src.model_restorer.model_restorer_interface import IModelRestorer
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.audio_features.types import AFTypes
from src.utils.files import Files

from src.utils.logger.logger_service import Logger

from src.definitions import frame_length, hop_length, sr


class ModelRestorer(IModelRestorer):
    def __init__(self):
        self.loger = Logger('ModelRestorer')
        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))
        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))
        self._experiment_step = ExperimentStep(experiment_id=0,
                                               af_strategy=AFStrategyFactory(sr=sr, frame_length=frame_length,
                                                                             hop_length=hop_length).create_strategy(
                                                   strategy_type=AFTypes.mfcc))
        self.files = Files()

    def restore_step(self, step_id: Optional[int] = None):
        best_step = self._experiment_step_model_service.get_best_step(step_id)
        self.loger.log(f"Restoring model from step {best_step.id}, with accuracy {best_step.accuracy_delta}")
        dataset_details = self._experiment_model_service.get_data_set_details(best_step.experiment_id)
        details = self._experiment_model_service.get_details(best_step.experiment_id)
        if DURATION != dataset_details.duration:
            raise Exception(f"Duration of step {best_step.id} is not equal to duration on definitions.py please change it to {dataset_details.duration}")

        af_strategy = AFStrategyFactory(sr=sr, frame_length=frame_length,
                                        hop_length=hop_length).create_strategy(dataset_details.af_type)
        data_set_importer = DataSetImporter(experiment_id=best_step.experiment_id, duration=dataset_details.duration,
                                            af_strategy=af_strategy)
        model_exporter = ModelExporter(af_strategy=af_strategy)
        asset_service = AssetsService(experiment_id=best_step.experiment_id)

        experiment_step = ExperimentStep(experiment_id=best_step.experiment_id, af_strategy=af_strategy)

        self.loger.log(f"Restoring model...")
        train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()
        mode, _ = experiment_step.prepare_step_model(step_id=best_step.id, data_sets=(train_ds, val_ds, test_ds),
                                                           epochs=details.epochs)
        self.loger.log(f"Model restored", color="green")
        path, mode_name = asset_service.get_models_path()
        export_path = self.files.join(path, mode_name)
        self.loger.log(f"Exporting model into {export_path}")
        model_exporter.export_model(mode, label_names, export_path)
        self.loger.log(f"Model saved", color="green")

    def restore_best_step(self):
        self.restore_step()
