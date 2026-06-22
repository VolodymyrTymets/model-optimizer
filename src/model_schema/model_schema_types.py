from abc import ABC
from enum import Enum
from typing import Sequence, Optional


class LayerType(Enum):
    Dense = 'Dense'
    Conv = 'Conv'
    GRU = 'GRU'


class ActivationType(Enum):
    Sigmoid = 'Sigmoid'
    tanh = 'tanh'
    ReLU = 'ReLU'


class RegularizerType(Enum):
    L2 = 'L2'
    L1 = 'L1'


class OptimizerType(Enum):
    Adam = 'Adam'
    AdamW = 'AdamW'
    Rmsprop = 'Rmsprop'


class LossType(Enum):
    MSE = 'MSE'
    BinaryCrossentropy = 'BinaryCrossentropy'


class ILayerSchema(ABC):
    type: LayerType
    units: int
    activation: ActivationType
    regularizer: Optional[RegularizerType]

class LayerSchema(ILayerSchema):
    def __init__(self, type: LayerType, units: int, activation: ActivationType, regularizer: Optional[RegularizerType] = None):
        self.type = type
        self.units = units
        self.activation = activation
        self.regularizer = regularizer


class IModelSchema(ABC):
    layers: Sequence[ILayerSchema]
    optimizer: OptimizerType
    loss: LossType


class ModelSchema(IModelSchema):
    def __init__(self, layers: Sequence[ILayerSchema], optimizer: OptimizerType, loss: LossType):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
