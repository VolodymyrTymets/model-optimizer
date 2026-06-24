from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class IModelRecordLabeler(ABC):
    @abstractmethod
    def label_records(self, model: tf.keras.Model, from_path: str) -> None:
        pass
