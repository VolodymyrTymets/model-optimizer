import tensorflow as tf

from abc import ABC, abstractmethod




class IModeTrainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
        pass
