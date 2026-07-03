import tensorflow as tf
from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import IModelSchema, ModelSchema, LayerSchema
from src.model_tuner.layer_tuner.layer_tuner import LayerTuner
from src.model_tuner.mode_tuner_interface import IModeTuner

from src.utils.logger.logger_service import Logger


class ModeTuner(IModeTuner):
    def __init__(self, details: IExperimentDetails, experiment_step: IExperimentStep):
        self._logger = Logger('ModeTuner')
        self._experiment_step = experiment_step
        self._details = details
        self.layer_tuner = LayerTuner(details, experiment_step)

    # first step: 1 layer, low units, sequential optimizer, loses
    def rare_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> IModelSchema:
        layers = self.layer_tuner.rare_tuning(data_sets)

        steps = []
        for optimizer in self._details.optimizer:
            for loss in self._details.loss:
                schema = ModelSchema(layers=layers, optimizer=optimizer, loss=loss)
                step = self._experiment_step.run(schema, data_sets, epochs=self._details.epochs)
                steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for rare_tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}", )
        return best_schema

    # second step: 2 layers, high units, sequential activations, regularization
    def layers_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                      schema: IModelSchema) -> IModelSchema:
        self._logger.log(f"Layers tuning started", color="green")
        steps = []
        # tuning layers indiviually
        schema.layers = []
        for layer in self._details.layers:
            self._logger.log(f"Layer {','.join(x.type.value for x in schema.layers)} tuning started", color="yellow")
            current_layer, best_step = self.layer_tuner.tuning(data_sets, schema, LayerSchema(layer, 0))
            schema.layers.append(current_layer)
            steps.append(best_step)
        # tuning layers together
        schema.layers = []
        for layer in self._details.layers:
            self._logger.log(f"Layer {''.join(x.type.value for x in schema.layers)} tuning started", color="yellow")
            current_layer, best_step = self.layer_tuner.tuning(data_sets, schema, LayerSchema(layer, 0), append_layers_together=True)
            schema.layers.append(current_layer)
            steps.append(best_step)

        self._logger.log(f"Layers tuning finished", color="green")
        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        self._logger.log(
            f"Best step for layers_tuning found on step {best_step.step} with accuracy {best_step.accuracy_delta}",
            color="green")
        self._logger.log(f"schema: {str(best_schema)}")
        return best_schema

    # third step: 3 ... todo: think about it, maybe argumentation, time, audio features
    def final_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                     schema: IModelSchema) -> IModelSchema:
        self._logger.log(f"todo:Final tuning started", color="yellow")
        self._logger.log(f"schema: {str(schema)}", )
        # todo: final tuning
        # for optimizer in self._details.optimizer:
        #     for loss in self._details.loss:
        #         schema = ModelSchema(layers=layers, optimizer=optimizer, loss=loss)
        #         step = self._experiment_step.run(schema, data_sets, epochs=self._details.epochs)
        #         steps.append(step)

        return schema

    def get_current_shema(self) -> IModelSchema:
        pass
