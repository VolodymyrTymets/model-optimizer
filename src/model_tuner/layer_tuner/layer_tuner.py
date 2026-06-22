from typing import Sequence

import tensorflow as tf

from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import ILayerSchema, LayerSchema, LayerType, ModelSchema
from src.model_tuner.layer_tuner.layer_tuner_interface import ILayerTuner
from src.utils.logger.logger_service import Logger


class LayerTuner(ILayerTuner):

    def __init__(self, details: IExperimentDetails, experiment_step: IExperimentStep):
        self._details = details
        self._experiment_step = experiment_step
        self._logger = Logger('LayerTuner')
        self.units_range = self.generate_rangers()

    def generate_rangers(self):
        ranges = []
        pow = 2
        while 2 ** pow <= self._details.units_range[-1]:
            if 2 ** pow > self._details.units_range[0]:
                ranges.append(2 ** pow)
            pow = pow + 1
        return ranges

    def rare_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> Sequence[ILayerSchema]:
        low_units = int(len(self.units_range) / 2)
        units = self.units_range[int(low_units / 2)]
        steps = []
        for activation in self._details.activation:
            step = self._experiment_step.run(ModelSchema(
                layers=[LayerSchema(self._details.layers[0], units, activation=activation)],
                optimizer=self._details.optimizer[0],
                loss=self._details.loss[0],
            ), train_data, test_data)
            steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)

        return best_schema.layers

    def tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset,
               schema: list[ILayerSchema]) -> list[ILayerSchema]:
        self._logger.log("todo: Layer tuning started", color="yellow")
        return schema
