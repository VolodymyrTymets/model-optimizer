import tensorflow as tf

from src.experiment.experiment_types import IExperimentDetails
from src.experiment.experiment_interface import IExperiment
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_builder.mode_builder_interface import IModeBuilder
from src.model_schema.model_schema_types import ModelSchema, LayerSchema
from src.model_trainer.mode_trainer_interface import IModeTrainer
from src.model_validator.mode_validator_interfcace import IModeValidator
from src.utils.logger.logger_service import ILogger


class Experiment(IExperiment):
    def __init__(self, details: IExperimentDetails, logger: ILogger, model_builder: IModeBuilder,
                 mode_trainer: IModeTrainer, mode_validator: IModeValidator, ):
        self._logger = logger
        self._model_builder = model_builder
        self._mode_trainer = mode_trainer
        self._mode_validator = mode_validator
        self._experiment_model_service = ExperimentModelService(logger)
        self._experiment_step_model_service = ExperimentStepModelService(logger)
        self._experiment_model = self._experiment_model_service.get_current_experiment(details)
        self._details = self._experiment_model_service.get_details(self._experiment_model)


    def run(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset):
        self._logger.log("Experiment started")
        step = 1
        # todo: optimize algorithm
        for optimizer in self._details.optimizer:
            for loss in self._details.loss:
                for layer in self._details.layers:
                    layers = []
                    for unit in self._details.units_range:
                        for activation in self._details.activation:
                            for regularizer in self._details.regularizer:
                                layers.append(LayerSchema(layer, unit, activation, regularizer))
                                schema = ModelSchema(
                                    layers=layers,
                                    optimizer=optimizer,
                                    loss=loss,
                                )
                                self._logger.log(f"[{step}]Experiment step started", color="blue")
                                existed_step = self._experiment_step_model_service.find(self._experiment_model.id, schema)
                                if existed_step is not None:
                                    self._logger.log(f"[{step}]Experiment step already exists", color="yellow")
                                    if existed_step.endAt is not None:
                                        step += 1
                                        continue
                                else:
                                    self._experiment_step_model_service.start_experiment_step(self._experiment_model.id, step, schema)
                                model = self._model_builder.build_model(schema)
                                model = self._mode_trainer.train(model, train_data)
                                record_acc, valid_acc = self._mode_validator.validate(model, test_data)
                                self._experiment_step_model_service.finish_experiment_step(self._experiment_model.id, schema, record_acc, valid_acc)
                                step += 1
                    layer = []

        self._experiment_model_service.finish_experiment(self._experiment_model.id)
        self._logger.log("Experiment finished")

