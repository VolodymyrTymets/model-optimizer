import tensorflow as tf

from src.definitions import EMULATE_MODE, labels
from src.model_builder.mode_builder_interface import IModeBuilder
from src.model_schema.model_schema_types import IModelSchema, ActivationType, ILayerSchema, LayerType, OptimizerType, \
    LossType
from src.utils.logger.logger_interface import ILogger


class ModeBuilder(IModeBuilder):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def _get_activation(self, activation: ActivationType):
        if activation is None:
            return None
        if activation.value == ActivationType.ReLU.value:
            return tf.nn.relu
        elif activation.value == ActivationType.Sigmoid.value:
            return tf.nn.sigmoid
        return None

    def _build_layer(self, layer: ILayerSchema, input_shape):
        activation = self._get_activation(layer.activation)
        if layer.type.value == LayerType.Dense.value:
            return [tf.keras.layers.Dense(units=layer.units, activation=activation)]
        if layer.type.value == LayerType.Conv.value:
            return [
                tf.keras.layers.Reshape((input_shape + (1,)), name="reshape_to_conv"),
                tf.keras.layers.Conv2D(layer.units, kernel_size=2, activation=activation),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Lambda(lambda x: tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x), name='reshape_after_conv')
            ]
        if layer.type.value == LayerType.GRU.value:
            layers = [
                # tf.keras.layers.Reshape((64, 43, -1)),
                tf.keras.layers.GRU(units=layer.units, activation=activation, return_sequences=True)]
            print('layers:', layers)
            return layers
        else:
            raise ValueError(f"Unsupported layer type: {layer.type}")

    def _get_optimizer(self, optimizer: OptimizerType, learning_rate=0.0001):
        if optimizer.value == OptimizerType.Adam.value:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.value == OptimizerType.AdamW.value:
            return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer.value == OptimizerType.Rmsprop.value:
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def _get_loss(self, loss: LossType):
        if loss.value == LossType.BinaryCrossentropy.value:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if loss.value == LossType.SparseCategoricalCrossentropy.value:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def build_model(self, schema: IModelSchema, train_ds: tf.data.Dataset) -> tf.keras.Model:
        self._logger.log(f"Building model schema: {str(schema)}", color="yellow")
        if EMULATE_MODE:
            return tf.keras.Sequential()

        input_shape = None
        for example, example_spect_labels in train_ds.take(1):
            input_shape = example.shape[1:]

        norm_layer = tf.keras.layers.Normalization(name='normalization')
        norm_layer.adapt(data=train_ds.map(
            map_func=lambda spec, label: spec))

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(norm_layer)
        for layer in schema.layers:
            for sublayer in self._build_layer(layer, input_shape):
                model.add(sublayer)
            model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))
        model.get_layer('normalization').adapt(train_ds.map(lambda x, label: x))


        model.compile(
            optimizer=self._get_optimizer(schema.optimizer),
            loss=self._get_loss(schema.loss),
            metrics=['accuracy'],
        )
        print('before summary:')
        model.summary()
        return model
