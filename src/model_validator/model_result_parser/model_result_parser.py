from typing import Literal, Union

import numpy as np
from src.definitions import FRAGMENT_LENGTH, labels
import tensorflow as tf
from src.utils.lists import pad_list

from src.model_validator.model_result_parser.model_result_parser_interface import IModelResultParser
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger


class ModelResultParser(IModelResultParser):
    def __init__(self, af_strategy: IAFStrategy):
        self.strategy = af_strategy
        self.logger = Logger('ModelResultParser')
        self.label_names = labels
        self.label_names.sort()

    def _get_af(self, x: np.ndarray) -> np.ndarray:
        return self.strategy.get_audio_feature(signal=pad_list(x, FRAGMENT_LENGTH))

    def _reshape(self, model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
        if x.shape != model.input_shape:
            return tf.reshape(x, (-1,) + x.shape)
        return x

    def _parse_in_memory_model(self, model: tf.keras.Model, x: np.ndarray) -> tuple[str, float]:
        result = model(tf.convert_to_tensor(x, dtype=tf.float32))
        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return class_names.numpy()[0].decode("utf-8"), result.numpy()[0][class_ids.numpy()[0]]

    def _parse_on_disk_model(self, model: tf.keras.Model, x: np.ndarray) -> tuple[str, float]:
        result = model(tf.convert_to_tensor(x, dtype=tf.float32))
        label = result['label_names'].numpy()[result['class_ids'].numpy()[0]]
        prediction = result['predictions'].numpy()[0][result['class_ids'].numpy()[0]]
        return label.decode("utf-8"), prediction

    def parse(self, model: tf.keras.Model, x: np.ndarray) -> tuple[str, float]:
        af = self._reshape(model, self._get_af(x))
        try:
            label, prediction = self._parse_on_disk_model(model, af)
            return label, prediction
        except Exception as e:
            try:
                label, prediction = self._parse_in_memory_model(model, af)
                return label, prediction
            except Exception as e:
                self.logger.error(e)

        return 'noise', 0