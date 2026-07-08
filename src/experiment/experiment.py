import tensorflow as tf

from src.assets_service.assets_service import AssetsService
from src.experiment.experiment_summarize_service.expirement_summarize_service import ExperimentSummarizeService
from src.experiment.experiment_types import IExperimentDetails, IExperimentDataSetDetails
from src.experiment.experiment_interface import IExperiment
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.experiment_step.experiment_step import ExperimentStep
from src.model_schema.model_schema_types import IModelSchema
from src.model_tuner.mode_tuner import ModeTuner

from src.utils.logger.logger_service import Logger
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class Experiment(IExperiment):
    def __init__(self, details: IExperimentDetails, data_set_details: IExperimentDataSetDetails,
                 af_strategy: IAFStrategy):
        self._logger = Logger('Experiment')

        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))

        self._experiment_model = self._experiment_model_service.get_current_experiment(details, data_set_details)
        self._experiment_step = ExperimentStep(self._experiment_model.id, af_strategy)
        self.model_tuner = ModeTuner(details, experiment_step=self._experiment_step)
        self.experiment_summary_service = ExperimentSummarizeService(self._experiment_model, af_strategy, AssetsService(
            experiment_id=self._experiment_model.id))

    def _finish_unfinished_steps(self):
        unfinished_steps = self._experiment_model_service.get_unfinished_steps(self._experiment_model.id)
        experiment_details = self._experiment_model_service.get_details(self._experiment_model.id)
        for unfinished_step in unfinished_steps:
            unfinished_step_schema = self._experiment_step.get_schema(unfinished_step)
            self._logger.log(f"Run unfinished step {unfinished_step.step}...", color="blue")
            self._experiment_step.run(unfinished_step_schema, experiment_details.epochs)
        self._logger.log("Unfinished steps finished", color="green")

    def get_experiment_id(self) -> int:
        return self._experiment_model.id

    def is_finished(self) -> bool:
        return self._experiment_model.endAt is not None

    def start(self) -> IModelSchema:

        if self._experiment_model.endAt is not None:
            self._logger.log("Experiment already finished", color="yellow")
            return self.experiment_summary_service.get_best_step_schema(experiment_id=self._experiment_model.id)

        self._logger.log("Experiment started for", self._experiment_model.id)

        schema = self.model_tuner.rare_tuning()
        schema = self.model_tuner.layers_tuning(schema)

        final_schema = self.model_tuner.final_tuning(schema)

        # run unfinished steps if any
        self._finish_unfinished_steps()

        return final_schema

    def finish(self):
        self._experiment_model_service.finish_experiment(self._experiment_model.id)
        self._logger.log("Experiment finished")

    def summarize(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], labels: list[str]):
        self.experiment_summary_service.summarize(data_sets, labels)
