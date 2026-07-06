import tensorflow as tf

from typing import Sequence

from src.database.schema import ExperimentStepModel
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

    def _get_best_units(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], schema: IModelSchema, units: list[int], unit_steps: list[ExperimentStepModel]) -> list[ExperimentStepModel]:
        if len(units) == 1:
            return unit_steps
        low_units = self._get_half(units, True)
        high_units = self._get_half(units, False)
        steps = []

        for current_units in [low_units, high_units]:
            count_of_units = self._get_mid(current_units)
            schema.layers[-1].units = count_of_units
            step = self._experiment_step.run(schema, data_sets, self._details.epochs)
            steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        next_units = low_units if best_schema.layers[-1].units in low_units else high_units
        unit_steps.append(best_step)
        return self._get_best_units(data_sets, best_schema, next_units, unit_steps)

    def get_best_settings(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], schema: IModelSchema,
               current_layer: ILayerSchema) -> ILayerSchema:
        options_schema = ModelSchema(
            layers=[LayerSchema(current_layer.type, self._get_mid(self.units_range))],
            optimizer=schema.optimizer,
            loss=schema.loss,
        )
        steps = [self._experiment_step.run(options_schema, data_sets, self._details.epochs)]
        for activation in self._details.activation:
            for regularizer in self._details.regularizer:
                options_schema.layers[-1].activation = activation
                options_schema.layers[-1].regularizer = regularizer
                steps.append(self._experiment_step.run(options_schema, data_sets, self._details.epochs))

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        return best_schema.layers[-1]


    def rare_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> Sequence[ILayerSchema]:
        low_units = self._get_half(self.units_range, True)
        units = self._get_mid(low_units)
        return [LayerSchema(self._details.layers[0], units)]

    def tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], schema: IModelSchema,
               current_layer: ILayerSchema, append_layers_together = False) -> tuple[ILayerSchema, ExperimentStepModel]:
        # step: 1 find best settings for current layer
        settings = self.get_best_settings(data_sets, schema, current_layer)
        new_layer = LayerSchema(current_layer.type, settings.units, settings.activation, settings.regularizer)
        layers = schema.layers + [new_layer] if append_layers_together else [new_layer]

        units_schema = ModelSchema(
            layers=layers,
            optimizer=schema.optimizer,
            loss=schema.loss,
        )
        units_steps = []
        # step: 2 find best units for current layer
        self._get_best_units(data_sets, units_schema, self.units_range, units_steps)
        best_step = self._experiment_step.get_best_step(units_steps)
        best_schema = self._experiment_step.get_schema(best_step)
        best_count_of_units = best_schema.layers[-1].units
        self._logger.log(f"    Best units for current layer {units_schema.layers[-1].type}: {best_count_of_units} found on step {best_step.step}", color="green")
        units_schema.layers[-1].units = best_count_of_units
        return units_schema.layers[-1], best_step
