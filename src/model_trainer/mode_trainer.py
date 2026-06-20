import time

import tensorflow as tf

from src.utils.logger.logger_interface import ILogger
from src.model_trainer.mode_trainer_interface import IModeTrainer


class ModeTrainer(IModeTrainer):
    def __init__(self, logger: ILogger,):
        self._logger = logger

    def train(self, model: tf.keras.Model, data: tf.data.Dataset) -> tf.keras.Model:
        self._logger.log("Training started")
        time.sleep(0.5)  # simulate training process
        self._logger.log("Training finished")
        return model
