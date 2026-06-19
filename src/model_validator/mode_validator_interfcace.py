from abc import ABC, abstractmethod

import tensorflow as tf


class IModeValidator(ABC):
    @abstractmethod
    def validate(self, model: tf.keras.Model, data: tf.data.Dataset) -> tuple[float, float]:
        pass
