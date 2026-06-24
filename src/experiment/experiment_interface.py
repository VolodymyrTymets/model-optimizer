import tensorflow as tf
from abc import ABC, abstractmethod
from src.model_schema.model_schema_types import IModelSchema


class IExperiment(ABC):
    @abstractmethod
    def start(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> IModelSchema:
        pass

    @abstractmethod
    def summarize(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]):
        pass

    @abstractmethod
    def finish(self):
        pass
