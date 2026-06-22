from typing import Sequence

import tensorflow as tf

from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import ILayerSchema, LayerSchema, LayerType, ModelSchema, IModelSchema
from src.model_tuner.layer_tuner.layer_tuner_interface import ILayerTuner
from src.utils.logger.logger_service import Logger


class LayerTuner(ILayerTuner):

    def __init__(self, details: IExperimentDetails, experiment_step: IExperimentStep):
        self._details = details
        self._experiment_step = experiment_step
        self._logger = Logger('LayerTuner')
        self.units_range: list[int] = self._generate_rangers()

    def _generate_rangers(self) -> list[int]:
        ranges = []
        pow = 2
        while 2 ** pow <= self._details.units_range[-1]:
            if 2 ** pow > self._details.units_range[0]:
                ranges.append(2 ** pow)
            pow = pow + 1
        return ranges

    def _get_half(self, arr: list[int], low: bool = True):
        mid = len(arr) // 2
        return  arr[:mid] if low else arr[mid:]

    def _get_mid(self, arr: list[int]):
        return arr[len(arr) // 2]

    def _get_best_units(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset, schema: IModelSchema, units: list[int]) -> int:
        if len(units) == 1:
            return schema.layers[-1].units
        low_units = self._get_half(units, True)
        high_units = self._get_half(units, False)
        steps = []

        for current_units in [low_units, high_units]:
            count_of_units = self._get_mid(current_units)
            schema.layers[-1].units = count_of_units
            step = self._experiment_step.run(schema, train_data, test_data)
            steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        next_units = low_units if best_schema.layers[-1].units in low_units else high_units
        return self._get_best_units(train_data, test_data, best_schema, next_units)


    def rare_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> Sequence[ILayerSchema]:
        low_units = self._get_half(self.units_range, True)
        units = self._get_mid(low_units)
        return [LayerSchema(self._details.layers[0], units)]

    def tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset, schema: IModelSchema,
               current_layer: ILayerSchema) -> ILayerSchema:

        modes_schema = ModelSchema(
            layers=schema.layers + [LayerSchema(current_layer.type, current_layer.units)],
            optimizer=schema.optimizer,
            loss=schema.loss,
        )

        # step: 1 find best units for current layer
        best_count_of_units = self._get_best_units(train_data, test_data, modes_schema, self.units_range)

        # step: 2 find best units for current layer
        modes_schema.layers[-1].units = best_count_of_units
        steps = [self._experiment_step.run(modes_schema, train_data, test_data)]

        for activation in self._details.activation:
            for regularizer in self._details.regularizer:
                modes_schema.layers[-1].activation = activation
                modes_schema.layers[-1].regularizer = regularizer
                steps.append(self._experiment_step.run(modes_schema, train_data, test_data))
        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        return best_schema.layers[-1]
