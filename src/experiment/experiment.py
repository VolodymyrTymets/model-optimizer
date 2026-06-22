import tensorflow as tf

from src.experiment.experiment_types import IExperimentDetails
from src.experiment.experiment_interface import IExperiment
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.experiment_step.experiment_step import ExperimentStep
from src.model_tuner.mode_tuner import ModeTuner

from src.utils.logger.logger_service import Logger


class Experiment(IExperiment):
    def __init__(self, details: IExperimentDetails):
        self._logger = Logger('Experiment')


        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))

        self._experiment_model = self._experiment_model_service.get_current_experiment(details)
        self._details = self._experiment_model_service.get_details(self._experiment_model)
        self.model_tuner = ModeTuner(details, ExperimentStep(self._experiment_model.id))


    # def run_old(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset):
    #     self._logger.log("Experiment started")
    #     self.current_step = 1
    #     # todo: optimize algorithm
    #     for optimizer in self._details.optimizer:
    #         for loss in self._details.loss:
    #             for layer in self._details.layers:
    #                 layers = []
    #                 for unit in self._details.units_range:
    #                     for activation in self._details.activation:
    #                         for regularizer in self._details.regularizer:
    #                             layers.append(LayerSchema(layer, unit, activation, regularizer))
    #                             schema = ModelSchema(
    #                                 layers=layers,
    #                                 optimizer=optimizer,
    #                                 loss=loss,
    #                             )
    #                             self._logger.log(f"[{self.current_step}]Experiment step started", color="blue")
    #                             existed_step = self._experiment_step_model_service.find(self._experiment_model.id,
    #                                                                                     schema)
    #                             if existed_step is not None:
    #                                 self._logger.log(f"[{self.current_step}]Experiment step already exists",
    #                                                  color="yellow")
    #                                 if existed_step.endAt is not None:
    #                                     self.current_step += 1
    #                                     continue
    #                             else:
    #                                 self._experiment_step_model_service.start_experiment_step(self._experiment_model.id,
    #                                                                                           step, schema)
    #                             model = self._model_builder.build_model(schema)
    #                             model = self._mode_trainer.train(model, train_data)
    #                             record_acc, valid_acc = self._mode_validator.validate(model, test_data)
    #                             self._experiment_step_model_service.finish_experiment_step(self._experiment_model.id,
    #                                                                                        schema, record_acc,
    #                                                                                        valid_acc)
    #                             self.current_step += 1
    #                 layer = []
    #
    #     self._experiment_model_service.finish_experiment(self._experiment_model.id)
    #     self._logger.log("Experiment finished")

    def run(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset):
        self._logger.log("Experiment started")

        schema = self.model_tuner.rare_tuning(test_data, train_data)
        for index, layer in enumerate(self._details.layers):
            schema = self.model_tuner.layers_tuning(test_data, train_data, schema, index)

        schema = self.model_tuner.final_tuning(test_data, train_data, schema)

        self._experiment_model_service.finish_experiment(self._experiment_model.id)
        self._logger.log("Experiment finished")
