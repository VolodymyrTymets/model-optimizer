from src.assets_service.assets_service import AssetsService
from src.data_set.data_set_cooker import DataSetCooker
from src.data_set.data_set_importer import DataSetImporter
from src.database.schema import ExperimentStepModel
from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_step.experiment_step_model_cooker import ExperimentStepModelCooker
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_schema.model_schema_types import IModelSchema, ModelSchema, LayerType, LayerSchema, ActivationType, \
    RegularizerType, OptimizerType, LossType
from src.model_trainer.mode_trainer import ModeTrainer
from src.model_validator.mode_validator import ModeValidator
from src.model_builder.mode_builder import ModeBuilder
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger


class ExperimentStep(IExperimentStep):
    def __init__(self, experiment_id: int, af_strategy: IAFStrategy):
        self.experiment_id = experiment_id
        self.assets_service = AssetsService(experiment_id)
        self._model_builder = ModeBuilder(logger=Logger('ModeBuilder'))
        self._mode_trainer = ModeTrainer(logger=Logger('ModeTrainer'))
        self._mode_validator = ModeValidator(logger=Logger('ModeValidator'), af_strategy=af_strategy)
        self._logger = Logger('ExperimentStep')
        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))
        self.experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))
        self.experiment_data_set_details = self.experiment_model_service.get_data_set_details(experiment_id)
        self.data_set_cooker = DataSetCooker(experiment_id=experiment_id, af_strategy=af_strategy)
        self.data_set_importer = DataSetImporter(experiment_id=experiment_id, af_strategy=af_strategy, duration=self.experiment_data_set_details.duration)
        self._experiment_step_model_cooker = ExperimentStepModelCooker(experiment_id=experiment_id, af_strategy=af_strategy)

    def get_schema(self, step: ExperimentStepModel) -> IModelSchema:
        shema = self._experiment_step_model_service.get_schema(step.id)
        layers = []
        for layer in shema.model_layers:
            regularizer = RegularizerType(layer.regularizer) if layer.regularizer is not None else None
            activation = ActivationType(layer.activation) if layer.activation is not None else None
            layers.append(LayerSchema(type=LayerType(layer.type), activation=activation, regularizer=regularizer, units=layer.units))

        return ModelSchema(layers=layers, optimizer=OptimizerType(shema.optimizer), loss=LossType(shema.loss))

    def get_best_step(self, steps: list[ExperimentStepModel]) -> ExperimentStepModel:
        return max(steps, key=lambda x: x.record_accuracy)

    def run(self, schema: IModelSchema, epochs: int) -> ExperimentStepModel:
        self.data_set_cooker.step_prepare(self.experiment_data_set_details.duration, self.experiment_data_set_details.argumentation_types)
        train_ds, val_ds, test_ds, _ = self.data_set_importer.import_data_set()
        step = self._experiment_step_model_service.find(self.experiment_id, schema)

        if step is None:
            latest_step = self._experiment_step_model_service.get_last_step(self.experiment_id) + 1
            step = self._experiment_step_model_service.start_experiment_step(self.experiment_id,  latest_step, schema)
        elif step.endAt is not None and step.accuracy_delta > 0:
            self._logger.log(f"[{step.step}]Experiment step already finished", color="yellow")
            return step
        self._logger.log(f"[{step.step}]Experiment step started", color="blue")

        try:
            model, history = self._experiment_step_model_cooker.cook_step_model(step_id=step.id, data_sets=(train_ds, val_ds, test_ds), epochs=epochs)
        except Exception as e:
            self._logger.log(f"[{step.step}]Experiment step training failed", color="red")
            self._logger.error(e)
            return step
        record_acc, valid_acc, _ = self._mode_validator.validate(model, test_ds, self.assets_service.get_validation_records_path())
        self._experiment_step_model_service.finish_experiment_step(self.experiment_id,
                                                                   schema, record_acc,
                                                                   valid_acc, history)
        return self._experiment_step_model_service.find(self.experiment_id, schema)
