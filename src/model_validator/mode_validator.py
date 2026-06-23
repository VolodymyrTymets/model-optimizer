import random

import tensorflow as tf

from src.definitions import EMULATE_MODE
from src.utils.logger.logger_interface import ILogger
from src.model_validator.mode_validator_interfcace import IModeValidator


class ModeValidator(IModeValidator):
    def __init__(self, logger: ILogger,):
        self._logger = logger

    def validate(self, model: tf.keras.Model, data: tf.data.Dataset) -> tuple[float, float]:
        if EMULATE_MODE:
            record_acc, validation_acc = random.randint(1, 100), random.randint(1, 100)  # simulate validation process
            return record_acc, validation_acc
        evaluation = model.evaluate(data)
        test_acc, test_loss = evaluation
        # todo: add validation dataset
        validation_acc = test_acc
        record_acc = test_acc
        return record_acc, validation_acc
