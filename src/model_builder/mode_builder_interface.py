import tensorflow as tf

from abc import ABC, abstractmethod
from typing import Sequence

from src.data_set.types import ArgumentationTypes

from src.model_schema.model_schema_types import IModelSchema


class IModeBuilder(ABC):
    @abstractmethod
    def build_model(self, schema: IModelSchema, train_ds: tf.data.Dataset,
                    argumentation_types: Sequence[ArgumentationTypes] = ()) -> tf.keras.Model:
        pass
