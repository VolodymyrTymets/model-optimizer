import tensorflow as tf
from tensorflow.python.data.ops.optional_ops import Optional

from src.model_builder.mode_builder_interface import IModeBuilder
from src.model_schema.model_schema_types import IModelSchema, ActivationType, ILayerSchema, LayerType
from src.utils.logger.logger_interface import ILogger


class ModeBuilder(IModeBuilder):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def  _get_activation(self, activation: ActivationType):
        if activation is None:
            return None
        if activation.value == ActivationType.ReLU.value:
            return tf.nn.relu
        elif activation.value == ActivationType.Sigmoid.value:
            return tf.nn.sigmoid
        return None

    def  _build_layer(self, layer: ILayerSchema):
        activation = self._get_activation(layer.activation)
        if layer.type.value == LayerType.Dense.value:
            return tf.keras.layers.Dense(units=layer.units, activation=activation)
        if layer.type.value == LayerType.Conv.value:
            return tf.keras.layers.Conv2D(layer.units, kernel_size=2, activation=activation)
        if layer.type.value == LayerType.GRU.value:
            return tf.keras.layers.GRU(units=layer.units, activation=activation)
        else:
            raise ValueError(f"Unsupported layer type: {layer.type}")

    def build_model(self, schema: IModelSchema) -> tf.keras.Model:
        self._logger.log(f"Building model schema: {str(schema)}", color="yellow")
        model = tf.keras.Sequential()
        for layer in schema.layers:
            model.add(self._build_layer(layer))
        # todo: implement
        #model.build(input_shape=(None, 1))
        return model
