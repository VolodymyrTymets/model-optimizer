import tensorflow as tf
from abc import ABC, abstractmethod



class IModeValidator(ABC):
    @abstractmethod
    def validate(self, model: tf.keras.Model, data: tf.data.Dataset, validation_records_path: str) -> tuple[float, float]:
        pass
