import time

import tensorflow as tf

from src.utils.logger.logger_interface import ILogger
from src.model_trainer.mode_trainer_interface import IModeTrainer


class ModeTrainer(IModeTrainer):
    def __init__(self, logger: ILogger,):
        self._logger = logger

    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> tf.keras.Model:
        time.sleep(0.5)  # simulate training process
        return model
