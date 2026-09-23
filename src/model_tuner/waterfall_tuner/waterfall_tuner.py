from typing import Literal, Sequence

from src.database.schema import ExperimentStepModel
from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import ILayerSchema, LayerSchema, LayerType, ModelSchema, IModelSchema
from src.model_tuner.waterfall_tuner.waterfall_tuner_interface import IWaterfallTuner
from src.utils.logger.logger_service import Logger


class WaterfallTuner(IWaterfallTuner):

    def __init__(self, details: IExperimentDetails, experiment_step: IExperimentStep):
        self._details = details
        self._experiment_step = experiment_step
        self._logger = Logger('WaterfallTuner')
        self.layers_count = 3

    # forms an array of powers of 2 of length <= self.layers_count, built by halving unit_count
    # down to (but not below) self._details.units_range[0]; example: if unit_count = 16 then
    # direction = 'up' gives [4,8,16], direction = 'down' gives [16,8,4]
    def _get_units_range(self, unit_count: int, direction: Literal['up', 'down']) -> Sequence[int]:
        units = [unit_count]
        while len(units) < self.layers_count:
            next_unit = units[-1] // 2
            if next_unit < self._details.units_range[0]:
                break
            units.append(next_unit)
        units.reverse()

        if direction == 'up':
            return [unit for unit in units if unit <= self._details.units_range[-1]]
        return list(reversed([unit for unit in units if unit >= self._details.units_range[0]]))

    def _waterfall_tuning(self, units: Sequence[int], shema: IModelSchema,
                          layer: ILayerSchema) -> list[ExperimentStepModel]:
        steps = []
        shema.layers = []
        for unit in units:
            shema.layers.append(LayerSchema(layer.type, unit, layer.activation, layer.regularizer))
            self._logger.log(f"Waterfall Layer {''.join(x.type.value for x in shema.layers)} tuning started",
                             color="yellow")
            step = self._experiment_step.run(shema, self._details.epochs)
            steps.append(step)
        return steps

    def tuning(self, schema: IModelSchema) -> IExperimentStep:
        tuning_shema = ModelSchema(layers=schema.layers, optimizer=schema.optimizer, loss=schema.loss)
        tuning_layer = schema.layers[-1]
        tuning_shema.layers = []

        # up tuning layers
        up_units = self._get_units_range(tuning_layer.units, 'up')
        self._logger.log(f"Up units: {up_units}")
        up_steps = self._waterfall_tuning(up_units, tuning_shema, tuning_layer)
        # down tuning layers
        down_units = self._get_units_range(tuning_layer.units, 'down')
        self._logger.log(f"Down units: {down_units}")
        down_steps = self._waterfall_tuning(down_units, tuning_shema, tuning_layer)

        # find best step
        best_step = self._experiment_step.get_best_step([*up_steps, *down_steps])
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for Waterfall tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}")
        return best_step
