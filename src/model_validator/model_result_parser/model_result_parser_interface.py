from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from src.utils.audio_features.types import AFTypes


class IModelResultParser(ABC):
    @abstractmethod
    def parse(self, model: tf.keras.Model, x: np.ndarray) -> tuple[str, float]:
        pass
    @abstractmethod
    def parse_settings(self, model) -> tuple[AFTypes, float]:
        pass
