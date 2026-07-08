import tensorflow as tf
import shutil
from src.data_set.data_set_importer import DataSetImporter
from src.experiment.experiment_step.experiment_step_model_cooker import ExperimentStepModelCooker
from src.utils.files import Files

from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_exporter.model_weights_exporter.model_weights_exporter import ModelWeightsExporter

from src.data_set.utils.data_set_splitter import DataSetFileWorker
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.logger.logger_service import Logger
from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import FRAGMENT_LENGTH, sr, frame_length, hop_length
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class LocalModelLoader:
    def __init__(self, af_strategy: IAFStrategy, assets_service: IAssetsService,
                 model_parser: ModelResultParser):
        self.assets_service = assets_service
        self.files = Files()
        self.logger = Logger('DataSetLocalFilter')
        self.model_parser = model_parser
        self.af_strategy = af_strategy

    def _get_af_strategy(self, model):
        try:
            af_type, _ = self.model_parser.parse_settings(model=model)
            if af_type and af_type.value != self.af_strategy.AFType.value:
                self.logger.log(f'Changing audio feature strategy to {af_type.value}', color='yellow')
                return AFStrategyFactory(sr=sr, frame_length=frame_length,
                                         hop_length=hop_length).create_strategy(strategy_type=af_type)
        except Exception as e:
            self.logger.log(f'Error in model settings parsing: {e}', color='red')

    def get_model(self, duration: float):
        model_path = self.files.join(self.assets_service.get_assets_path(), 'models', f'model_{duration}')
        if not self.files.is_exist(model_path):
            return None, None
        model = tf.keras.models.load_model(model_path)

        return model, self._get_af_strategy(model)

    def is_filtered(self, db_path: str):
        return self.files.is_exist(self.files.join(db_path, '__filtered__'))

    def finish(self, db_path: str):
        self.files.create_folder(self.files.join(db_path, '__filtered__'))



class InMemoryModelLoader:
    def __init__(self, experiment_id: int, af_strategy: IAFStrategy, assets_service: IAssetsService, ):
        self.experiment_id = experiment_id
        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))
        self._model_step = ExperimentStepModelService(Logger('ExperimentStepModelService'))

        self._experiment_step_model_cooker = ExperimentStepModelCooker(experiment_id, af_strategy)
        self.model_weights_service = ModelWeightsExporter(assets_service)
        self.af_strategy = af_strategy
        self.files = Files()
        self.logger = Logger('DataSetInMemoryFilter')

    def _get_best_step(self):
        return self._experiment_step_model_service.get_best_step(self.experiment_id)

    def is_filtered(self, db_path: str):
        best_step = self._get_best_step()
        if best_step is None:
            self.logger.log('No best step found', color='red')
            return False
        path = self.files.join(db_path, '__filtered__')
        if self.files.is_exist(path):
            if not self.files.is_exist(self.files.join(path,"accuracy.txt")):
                self.logger.log(f'No accuracy file found for {db_path}', color='red')
                return False
            with open(self.files.join(path,"accuracy.txt"), "r", encoding="utf-8") as file:
                content = file.read()
                return best_step.accuracy_delta <= float(content.strip())
        return False

    def finish(self, db_path: str):
        best_step = self._get_best_step()
        path = self.files.join(db_path, '__filtered__')
        self.files.create_folder(path)
        with open(self.files.join(path,"accuracy.txt"), "w", encoding="utf-8") as file:
            file.write(str(best_step.accuracy_delta) + "\n")

    def get_model(self, duration: float):
        data_set_importer = DataSetImporter(experiment_id=self.experiment_id, duration=duration,
                                            af_strategy=self.af_strategy)
        train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()
        best_step = self._get_best_step()
        if best_step is None:
            return None, None
        self.logger.log(f'Loading model from step {best_step.step} with ac {best_step.accuracy_delta}', color='blue')
        model, _ = self._experiment_step_model_cooker.cook_step_model(step_id=best_step.id, data_sets=(train_ds, val_ds, test_ds), epochs=100)
        return model, None


class DataSetFilter(DataSetFileWorker):
    def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str],
                 assets_service: IAssetsService, af_strategy: IAFStrategy, experiment_id: int, localModel=False):
        super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
        self.assets_service = assets_service
        self.experiment_id = experiment_id
        self.af_strategy = af_strategy
        self.model_parser = ModelResultParser(af_strategy=self.af_strategy)
        self.logger = Logger('DataSetFilter')
        self.except_sets = []
        self.except_labels = []
        self.assets_service = assets_service
        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))
        self.model_loader = LocalModelLoader(af_strategy=self.af_strategy,
                                             assets_service=self.assets_service,
                                             model_parser=self.model_parser) if localModel else InMemoryModelLoader(
            af_strategy=self.af_strategy, assets_service=self.assets_service, experiment_id=self.experiment_id)

    def is_filtered(self):
        return self.model_loader.is_filtered(self.out_path)

    def finish(self):
        self.logger.log('Finishing', color='blue')
        self.model_loader.finish(self.out_path)

    def filter(self, duration: float) -> bool:
        self.logger.log('Start filtering', color='blue')
        if self.model_loader.is_filtered(self.out_path):
            self.logger.log('Filtering already done. Skipping.', color='blue')
            return False

        model, filter_af_strategy = self.model_loader.get_model(duration)
        if not model:
            self.logger.log('Model not found. Filtering skipped.', color='red')
            return False
        if filter_af_strategy is not None:
            self.model_parser = ModelResultParser(af_strategy=filter_af_strategy)
        is_filtered = False
        for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
            if set_name in self.except_sets:
                continue
            if label in self.except_labels:
                continue
            if len(signal) >= FRAGMENT_LENGTH:
                signal_label, _ = self.model_parser.parse(model=model, x=signal)
                if label != signal_label:
                    out_folder = self.files.join(self.out_path, set_name, signal_label)
                    self.files.create_folder(out_folder)
                    to_path = self.files.join(out_folder, file)
                    from_path = self.files.join(path, file)
                    self.logger.log(f'moving {from_path} to {to_path}', color='yellow')
                    shutil.move(from_path, to_path)
                    is_filtered = True

        self.logger.log('End filtering', color='blue')
        self.finish()
        return is_filtered
