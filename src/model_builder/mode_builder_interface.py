from abc import ABC, abstractmethod

import tensorflow as tf
from src.model_schema.model_schema_types import IModelSchema


class IModeBuilder(ABC):
    @abstractmethod
    def build_model(self, schema: IModelSchema) -> tf.keras.Model:
        pass
