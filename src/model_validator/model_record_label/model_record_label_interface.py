import tensorflow as tf
from abc import ABC, abstractmethod



class IModelRecordLabeler(ABC):
    @abstractmethod
    def label_records(self, model: tf.keras.Model, from_path: str) -> None:
        pass
