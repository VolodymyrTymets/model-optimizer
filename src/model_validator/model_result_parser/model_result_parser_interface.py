from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class IModelResultParser(ABC):
    @abstractmethod
    def parse(self, model: tf.keras.Model, x: np.ndarray) -> tuple[str, float]:
        pass
