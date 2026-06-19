from abc import ABC, abstractmethod

import tensorflow as tf
from src.model_schema.model_schema_types import IModelSchema, ActivationType, ILayerSchema, LayerType
from src.utils.logger.logger_interface import ILogger


class ModeBuilder(ABC):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def  _get_activation(self, activation: ActivationType):
        if activation.value == ActivationType.ReLU.value:
            return tf.nn.relu
        elif activation.value == ActivationType.Sigmoid.value:
            return tf.nn.sigmoid
        return None

    def  _build_layer(self, layer: ILayerSchema):
        self._logger.log(f"Layer: {layer.type.value}, Units: {layer.units}, Activation: {layer.activation}", color="blue")
        activation = self._get_activation(layer.activation)
        if layer.type.value == LayerType.Dense.value:
            return tf.keras.layers.Dense(units=layer.units, activation=activation)
        else:
            raise ValueError(f"Unsupported layer type: {layer.type}")

    def build_model(self, schema: IModelSchema) -> tf.keras.Model:
        self._logger.log(f"Building model:", color="blue")
        model = tf.keras.Sequential()
        for layer in schema.layers:
            model.add(self._build_layer(layer))
        self._logger.log(f"Optimizer: {schema.optimizer.value}", color="blue")
        self._logger.log(f"Loss: {schema.loss.value}", color="blue")
        # todo: implement
        model.build(input_shape=(None, 1))
        return model
