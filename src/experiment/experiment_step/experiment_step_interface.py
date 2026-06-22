from abc import ABC, abstractmethod

import tensorflow as tf

from src.database.schema import ExperimentStepModel, ModelSchemaModel
from src.model_schema.model_schema_types import IModelSchema


class IExperimentStep(ABC):
    @abstractmethod
    def run(self, schema: IModelSchema, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> ExperimentStepModel:
        pass

    @abstractmethod
    def get_schema(self, step: ExperimentStepModel) -> IModelSchema:
        pass

    @abstractmethod
    def get_best_step(self, steps: list[ExperimentStepModel]) -> ExperimentStepModel:
        pass
