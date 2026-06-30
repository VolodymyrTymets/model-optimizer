from abc import ABC
from typing import Sequence

from src.data_set.types import ArgumentationTypes
from src.model_schema.model_schema_types import LayerType, OptimizerType, LossType, RegularizerType, ActivationType
from src.utils.audio_features.types import AFTypes


class IExperimentDetails(ABC):
    layers: Sequence[LayerType]
    activation: Sequence[ActivationType]
    optimizer: Sequence[OptimizerType]
    regularizer: Sequence[RegularizerType]
    loss: Sequence[LossType]
    epochs: int
    batch_size: int
    units_range: Sequence[int]

class IExperimentDataSetDetails(ABC):
    duration: float
    labels: list[str]
    argumentation_types: list[ArgumentationTypes]
    af_type: AFTypes

class ExperimentDetails(IExperimentDetails):
    def __init__(self, layers: Sequence[LayerType], activation: Sequence[ActivationType],
                 optimizer: Sequence[OptimizerType], regularizer: Sequence[RegularizerType], loss: Sequence[LossType],
                 epochs: int, batch_size: int, units_range: Sequence[int]):
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.units_range = units_range

class ExperimentDataSetDetails(IExperimentDataSetDetails):
    def __init__(self, duration: float, labels: list[str], argumentation_types: list[ArgumentationTypes], af_type: AFTypes):
        self.duration = duration
        self.labels = labels
        self.argumentation_types = argumentation_types
        self.af_type = af_type

