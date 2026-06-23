from abc import ABC, abstractmethod
import tensorflow as tf
from src.model_schema.model_schema_types import IModelSchema


class IModeTuner(ABC):
    # first step: 1 layer, low units, sequential optimizer, loses
    @abstractmethod
    def rare_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> IModelSchema:
        pass

    # second step: 2 layers, high units, sequential activations, regularization
    @abstractmethod
    def layers_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                      schema: IModelSchema) -> IModelSchema:
        pass

    # third step: 3 ... todo: think about it, maybe argumentation, time, audio features
    @abstractmethod
    def final_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                     schema: IModelSchema) -> IModelSchema:
        pass

    @abstractmethod
    def get_current_shema(self) -> IModelSchema:
        pass
