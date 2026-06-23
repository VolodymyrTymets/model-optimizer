from abc import ABC, abstractmethod

import tensorflow as tf
from src.model_schema.model_schema_types import ILayerSchema, IModelSchema


class ILayerTuner(ABC):
    @abstractmethod
    def rare_tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> list[ILayerSchema]:
        pass

    @abstractmethod
    def tuning(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], schema: IModelSchema, current_layer: ILayerSchema) -> ILayerSchema:
        pass
