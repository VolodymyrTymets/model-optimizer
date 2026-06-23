import tensorflow as tf

from src.experiment.experiment_types import IExperimentDetails
from src.experiment.experiment_interface import IExperiment
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.experiment_step.experiment_step import ExperimentStep
from src.model_schema.model_schema_types import IModelSchema
from src.model_tuner.mode_tuner import ModeTuner

from src.utils.logger.logger_service import Logger


class Experiment(IExperiment):
    def __init__(self, details: IExperimentDetails):
        self._logger = Logger('Experiment')


        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))

        self._experiment_model = self._experiment_model_service.get_current_experiment(details)
        self._details = self._experiment_model_service.get_details(self._experiment_model)
        self.model_tuner = ModeTuner(details, ExperimentStep(self._experiment_model.id))

    def get_experiment_id(self) -> int:
        return self._experiment_model.id

    def start(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> IModelSchema:
        self._logger.log("Experiment started")

        schema = self.model_tuner.rare_tuning(data_sets)
        schema = self.model_tuner.layers_tuning(data_sets, schema)

        # todo: return schema with best settings
        return self.model_tuner.final_tuning(data_sets, schema)

    def finish(self):
        self._experiment_model_service.finish_experiment(self._experiment_model.id)
        self._logger.log("Experiment finished")
