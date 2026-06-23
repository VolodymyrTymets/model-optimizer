from abc import ABC, abstractmethod

import tensorflow as tf
from src.model_schema.model_schema_types import ILayerSchema, IModelSchema


class ILayerTuner(ABC):
    @abstractmethod
    def rare_tuning(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset) -> list[ILayerSchema]:
        pass

    @abstractmethod
    def tuning(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset, schema: IModelSchema, current_layer: ILayerSchema) -> ILayerSchema:
        pass
