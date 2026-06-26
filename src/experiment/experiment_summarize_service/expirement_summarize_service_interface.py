import tensorflow as tf
from abc import ABC, abstractmethod

from src.model_schema.model_schema_types import IModelSchema


class IExperimentSummarizeService(ABC):
    @abstractmethod
    def summarize(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], labels: list[str]) -> None:
        pass

    @abstractmethod
    def get_best_step_schema(self, experiment_id: int) -> IModelSchema:
        pass