import tensorflow as tf
from abc import ABC, abstractmethod



class IExperimentSummarizeService(ABC):
    @abstractmethod
    def summarize(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], labels: list[str]) -> None:
        pass