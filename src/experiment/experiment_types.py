from abc import ABC
from typing import Sequence, Protocol

from src.model_schema.model_schema_types import LayerType, OptimizerType, LossType, RegularizerType, ActivationType


class IExperimentDetails(ABC):
    layers: Sequence[LayerType]
    activation: Sequence[ActivationType]
    optimizer: Sequence[OptimizerType]
    regularizer: Sequence[RegularizerType]
    loss: Sequence[LossType]
    epochs: int
    batch_size: int
    units_range: Sequence[int]


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
