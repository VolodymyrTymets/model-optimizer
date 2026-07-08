import tensorflow as tf

from src.assets_service.assets_service import AssetsService
from src.database.schema import ExperimentStepModel
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_schema.model_schema_types import IModelSchema, ModelSchema, LayerType, LayerSchema, ActivationType, \
    RegularizerType, OptimizerType, LossType
from src.model_trainer.mode_trainer import ModeTrainer
from src.model_validator.mode_validator import ModeValidator
from src.model_builder.mode_builder import ModeBuilder
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger
from src.model_exporter.model_weights_exporter.model_weights_exporter import ModelWeightsExporter


class ExperimentStepModelCooker:
    def __init__(self, experiment_id: int, af_strategy: IAFStrategy):
        self.experiment_id = experiment_id
        self.assets_service = AssetsService(experiment_id)
        self._model_builder = ModeBuilder(logger=Logger('ModeBuilder'))
        self._mode_trainer = ModeTrainer(logger=Logger('ModeTrainer'))
        self._mode_validator = ModeValidator(logger=Logger('ModeValidator'), af_strategy=af_strategy)
        self._model_weights_service = ModelWeightsExporter(self.assets_service)
        self._logger = Logger('ExperimentStep')
        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))

    def get_schema(self, step: ExperimentStepModel) -> IModelSchema:
        shema = self._experiment_step_model_service.get_schema(step.id)
        layers = []
        for layer in shema.model_layers:
            regularizer = RegularizerType(layer.regularizer) if layer.regularizer is not None else None
            activation = ActivationType(layer.activation) if layer.activation is not None else None
            layers.append(LayerSchema(type=LayerType(layer.type), activation=activation, regularizer=regularizer, units=layer.units))

        return ModelSchema(layers=layers, optimizer=OptimizerType(shema.optimizer), loss=LossType(shema.loss))

    def cook_step_model(self, step_id: int, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], epochs: int) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        train_ds, val_ds, test_ds = data_sets
        best_step = self._experiment_step_model_service.get_step(step_id=step_id)
        best_schema = self.get_schema(step=best_step)
        model = self._model_builder.build_model(best_schema, train_ds)
        model = self._model_weights_service.import_weights(model, best_step.step)
        model, history = self._mode_trainer.train(model, train_ds, val_ds, epochs)
        return model, history

