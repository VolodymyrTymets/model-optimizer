import tensorflow as tf

from src.database.schema import ExperimentStepModel, ModelSchemaModel
from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_schema.model_schema_types import IModelSchema, ModelSchema, LayerType, LayerSchema, ActivationType, \
    RegularizerType, OptimizerType, LossType
from src.model_trainer.mode_trainer import ModeTrainer
from src.model_validator.mode_validator import ModeValidator
from src.model_builder.mode_builder import ModeBuilder
from src.utils.logger.logger_service import Logger


class ExperimentStep(IExperimentStep):
    def __init__(self, experiment_id: int):
        self.experiment_id = experiment_id
        self._model_builder = ModeBuilder(logger=Logger('ModeBuilder'))
        self._mode_trainer = ModeTrainer(logger=Logger('ModeTrainer'))
        self._mode_validator = ModeValidator(logger=Logger('ModeValidator'))
        self._logger = Logger('ExperimentStep')
        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))

    def get_schema(self, step: ExperimentStepModel) -> IModelSchema:
        shema = self._experiment_step_model_service.get_schema(step.id)
        layers = []
        for layer in shema.model_layers:
            regularizer = RegularizerType(layer.regularizer) if layer.regularizer is not None else None
            layers.append(LayerSchema(type=LayerType(layer.type), activation=ActivationType(layer.activation), regularizer=regularizer, units=layer.units))

        return ModelSchema(layers=layers, optimizer=OptimizerType(shema.optimizer), loss=LossType(shema.loss))

    def get_best_step(self, steps: list[ExperimentStepModel]) -> ExperimentStepModel:
        return max(steps, key=lambda x: x.record_accuracy)

    def run(self, schema: IModelSchema, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> ExperimentStepModel:
        existed_step = self._experiment_step_model_service.find(self.experiment_id, schema)
        step = existed_step.step + 1 if existed_step is not None else 1
        self._logger.log(f"[{step}]Experiment step started", color="blue")
        if existed_step is not None:
            self._logger.log(f"[{step}]Experiment step already exists", color="yellow")
            if existed_step.endAt is not None:
                return existed_step
        else:
            existed_step = self._experiment_step_model_service.start_experiment_step(self.experiment_id,
                                                                                     step, schema)
        model = self._model_builder.build_model(schema)
        model = self._mode_trainer.train(model, train_data)
        record_acc, valid_acc = self._mode_validator.validate(model, test_data)
        self._experiment_step_model_service.finish_experiment_step(self.experiment_id,
                                                                   schema, record_acc,
                                                                   valid_acc)
        return existed_step
