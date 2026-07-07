import tensorflow as tf
from abc import ABC, abstractmethod


class IModelRestorer(ABC):
    @abstractmethod
    def restore_step(self, step_id: int):
        pass
    @abstractmethod
    def restore_best_step(self) -> tf.keras.Model:
        pass