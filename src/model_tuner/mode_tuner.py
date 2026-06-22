import tensorflow as tf
from src.experiment.experiment_step.experiment_step_interface import IExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.model_schema.model_schema_types import IModelSchema, ModelSchema
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
    def rare_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> IModelSchema:
        layers = self.layer_tuner.rare_tuning(train_data, test_data)

        steps = []
        for optimizer in self._details.optimizer:
            for loss in self._details.loss:
                schema = ModelSchema(layers=layers, optimizer=optimizer, loss=loss)
                step = self._experiment_step.run(schema, train_data, test_data)
                steps.append(step)

        best_step = self._experiment_step.get_best_step(steps)
        best_schema = self._experiment_step.get_schema(best_step)
        return best_schema

    # second step: 2 layers, high units, sequential activations, regularization
    def layers_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset, schema: IModelSchema,
                      layer: int) -> IModelSchema:
        self._logger.log(f"todo: Layer {layer} tuning started", color="yellow")
        return schema

    # third step: 3 ... todo: think about it, maybe argumentation, time, audio features
    def final_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset,
                     schema: IModelSchema) -> IModelSchema:
        self._logger.log(f"todo:Final tuning started", color="yellow")
        return schema

    def get_current_shema(self) -> IModelSchema:
        pass
