from abc import ABC, abstractmethod

import tensorflow as tf
from src.model_schema.model_schema_types import ILayerSchema


class ILayerTuner(ABC):
    @abstractmethod
    def rare_tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> list[ILayerSchema]:
        pass

    @abstractmethod
    def tuning(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset,
               layers: list[ILayerSchema]) -> list[ILayerSchema]:
        pass
