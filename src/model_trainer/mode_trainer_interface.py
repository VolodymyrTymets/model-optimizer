from abc import ABC, abstractmethod

import tensorflow as tf


class IModeTrainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.Model, data: tf.data.Dataset) -> tf.keras.Model:
        pass
