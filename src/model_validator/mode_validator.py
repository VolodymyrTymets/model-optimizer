import random

import tensorflow as tf

from src.utils.logger.logger_interface import ILogger
from src.model_validator.mode_validator_interfcace import IModeValidator


class ModeValidator(IModeValidator):
    def __init__(self, logger: ILogger,):
        self._logger = logger

    def validate(self, model: tf.keras.Model, data: tf.data.Dataset) -> tuple[float, float]:
        # self._logger.log("Validation started")
        record_acc, validation_acc = random.randint(1, 100), random.randint(1, 100)  # simulate validation process
        # self._logger.log(f"Validation finished: {record_acc}, {validation_acc}")
        return record_acc, validation_acc
