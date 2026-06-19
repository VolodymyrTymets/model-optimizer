import tensorflow as tf

from src.experiment.experiment_types import IExperimentDetails
from src.experiment.experiment_interface import IExperiment
from src.experiment.model.experiment_model_service import ExperimentModelService
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
        self._experiment_model = self._experiment_model_service.get_current_experiment(details)
        self._details = self._experiment_model_service.get_details(self._experiment_model)

    def get_schema(self):
        schema = ModelSchema(
            layers=[LayerSchema(
                type=self._details.layers[0],
                units=self._details.units_range[0],
                activation=self._details.activation[0],
                regularizer=self._details.regularizer[0],
            )],
            optimizer=self._details.optimizer[0],
            loss=self._details.loss[0],
        )
        return schema

    def run(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset):
        self._logger.log("Experiment started")


        model = self._model_builder.build_model(self.get_schema())
        self._mode_trainer.train(model, train_data)
        self._mode_validator.validate(model, test_data)
        self._experiment_model_service.finish_experiment(self._experiment_model.id)
        self._logger.log("Experiment finished")

