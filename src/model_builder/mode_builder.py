from typing import Sequence

import tensorflow as tf

from src.data_set.types import ArgumentationTypes
from src.definitions import EMULATE_MODE, labels, DATA_SET_TYPE
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

    def _build_layer(self, layer: ILayerSchema, input_shape, current_shape):
        activation = self._get_activation(layer.activation)
        if layer.type.value == LayerType.Dense.value:
            return [tf.keras.layers.Dense(units=layer.units, activation=activation)]
        if layer.type.value == LayerType.Conv.value:
            layers = []
            # audio features are (T, F) and need a channel axis, images are already (H, W, C)
            if len(input_shape) == 2:
                layers.append(tf.keras.layers.Reshape((input_shape + (1,)), name="reshape_to_conv"))
            layers += [
                tf.keras.layers.Conv2D(layer.units, kernel_size=3, padding='same', activation=activation),
                tf.keras.layers.MaxPool2D(),
                # tf.keras.layers.Lambda(lambda x: tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x), name='reshape_after_conv')
            ]
            return layers
        if layer.type.value == LayerType.GRU.value:
            layers = []
            # GRU needs (time, features): images (H, W, C) are read as H time steps of W*C features
            if len(current_shape) == 3:
                layers.append(tf.keras.layers.Reshape((current_shape[0], current_shape[1] * current_shape[2]),
                                                      name="reshape_to_gru"))
            layers.append(tf.keras.layers.GRU(units=layer.units, activation=activation, return_sequences=True))
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
            return tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        if loss.value == LossType.SparseCategoricalCrossentropy.value:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def _get_augmentation_layers(self, argumentation_types: Sequence[ArgumentationTypes]):
        layers = []
        for argumentation_type in argumentation_types:
            if argumentation_type == ArgumentationTypes.RandomFlip:
                layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
            elif argumentation_type == ArgumentationTypes.RandomRotation:
                layers.append(tf.keras.layers.RandomRotation(0.1))
            elif argumentation_type == ArgumentationTypes.RandomZoom:
                layers.append(tf.keras.layers.RandomZoom(0.1))
        return layers

    def _get_normalization_layers(self, train_ds: tf.data.Dataset):
        if DATA_SET_TYPE == 'image':
            # pixels are already bounded to 0..255, so a fixed rescale is enough
            return [tf.keras.layers.Rescaling(1. / 255)]
        norm_layer = tf.keras.layers.Normalization(name='normalization')
        norm_layer.adapt(data=train_ds.map(lambda spec, label: spec))
        return [norm_layer]

    def build_model(self, schema: IModelSchema, train_ds: tf.data.Dataset,
                    argumentation_types: Sequence[ArgumentationTypes] = ()) -> tf.keras.Model:
        self._logger.log(f"Building model schema: {str(schema)}", color="yellow")
        if EMULATE_MODE:
            return tf.keras.Sequential()

        input_shape = None
        for example, example_spect_labels in train_ds.take(1):
            input_shape = example.shape[1:]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        # random augmentation layers are only active while training
        if DATA_SET_TYPE == 'image':
            for augmentation_layer in self._get_augmentation_layers(argumentation_types):
                model.add(augmentation_layer)
        for normalization_layer in self._get_normalization_layers(train_ds):
            model.add(normalization_layer)
        for layer in schema.layers:
            for sublayer in self._build_layer(layer, input_shape, model.output_shape[1:]):
                model.add(sublayer)
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))

        model.compile(
            optimizer=self._get_optimizer(schema.optimizer),
            loss=self._get_loss(schema.loss),
            metrics=['accuracy'],
        )
        model.summary()
        return model
