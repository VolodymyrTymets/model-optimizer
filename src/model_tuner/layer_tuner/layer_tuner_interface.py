import tensorflow as tf

from abc import ABC, abstractmethod

from src.experiment.experiment_step.experiment_step_interface import ExperimentStepModel
from src.model_schema.model_schema_types import ILayerSchema, IModelSchema


class ILayerTuner(ABC):
    @abstractmethod
    def rare_tuning(self) -> list[ILayerSchema]:
        pass

    @abstractmethod
    def tuning(self, schema: IModelSchema, current_layer: ILayerSchema) -> tuple[ILayerSchema, ExperimentStepModel]:
        pass
