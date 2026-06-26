import tensorflow as tf
from abc import ABC, abstractmethod


class IModelWeightsExporter(ABC):
    @abstractmethod
    def export_weights(self, model, step: int):
        pass
    @abstractmethod
    def import_weights(self, model, step: int) -> tf.keras.Model:
        pass