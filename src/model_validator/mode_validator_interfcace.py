from abc import ABC, abstractmethod

import tensorflow as tf



class IModeValidator(ABC):
    @abstractmethod
    def validate(self, model: tf.keras.Model, data: tf.data.Dataset, validation_records_path: str) -> tuple[float, float]:
        pass
