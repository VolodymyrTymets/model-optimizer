from abc import ABC, abstractmethod

from src.experiment.experiment_step.experiment_step_interface import ExperimentStepModel
from src.model_schema.model_schema_types import ILayerSchema, IModelSchema


class IWaterfallTuner(ABC):
    @abstractmethod
    def tuning(self, schema: IModelSchema) -> ILayerSchema:
        pass
