from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import IModelSchema, ModelSchema, LayerSchema
from src.model_tuner.layer_tuner.layer_tuner import LayerTuner
from src.model_tuner.mode_tuner_interface import IModeTuner
from src.model_tuner.waterfall_tuner.waterfall_tuner import WaterfallTuner

from src.utils.logger.logger_service import Logger


class ModeTuner(IModeTuner):
    def __init__(self, details: IExperimentDetails, experiment_step: IExperimentStep):
        self._logger = Logger('ModeTuner')
        self._experiment_step = experiment_step
        self._details = details
        self.layer_tuner = LayerTuner(details, experiment_step)
        self.waterfall_tuning = WaterfallTuner(details, experiment_step)

    # first step: 1 layer, low units, sequential optimizer, loses
    def rare_tuning(self) -> IModelSchema:
        layers = self.layer_tuner.rare_tuning()

        steps = []
        for optimizer in self._details.optimizer:
            for loss in self._details.loss:
                schema = ModelSchema(layers=layers, optimizer=optimizer, loss=loss)
                step = self._experiment_step.run(schema, epochs = self._details.epochs)
                steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for rare_tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}", )
        return best_schema

    # second step: 2 layers, high units, sequential activations, regularization
    def layers_tuning(self, schema: IModelSchema) -> IModelSchema:
        self._logger.log(f"Layers tuning started", color="green")
        steps = []
        waterfall_steps = []
        # tuning layers individually
        schema.layers = []
        for layer in self._details.layers:
            self._logger.log(f"Layer {','.join(x.type.value for x in schema.layers)} tuning started", color="yellow")
            current_layer, best_step = self.layer_tuner.tuning(schema, LayerSchema(layer, 0))
            steps.append(best_step)

            individua_layer_best_schema = self._experiment_step.get_schema(best_step)
            waterfall_layer_step = self.waterfall_tuning.tuning(individua_layer_best_schema)
            waterfall_steps.append(waterfall_layer_step)

        # tuning layers together
        together_steps = []
        for index, layer in enumerate(self._details.layers):
            if index == 0:
                continue
            waterfall = waterfall_steps[index - 1]
            waterfall_step_schema = self._experiment_step.get_schema(waterfall)

            self._logger.log(f"Layer {''.join(x.type.value for x in schema.layers)} tuning started", color="yellow")
            current_layer, best_step = self.layer_tuner.tuning(waterfall_step_schema, LayerSchema(layer, 0),
                                                               append_layers_together=True)
            together_steps.append(best_step)

        self._logger.log(f"Layers tuning finished", color="green")
        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for layers_tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}")
        return best_schema

    # third step: 3 final check optimizer, loss,
    def final_tuning(self, schema: IModelSchema) -> IModelSchema:
        steps = []
        for optimizer in self._details.optimizer:
            for loss in self._details.loss:
                schema.optimizer = optimizer
                schema.loss = loss
                step = self._experiment_step.run(schema, epochs=self._details.epochs)
                steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for final_tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}")
        return best_schema

    def get_current_shema(self) -> IModelSchema:
        pass
