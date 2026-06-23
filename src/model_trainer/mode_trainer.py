import time

import tensorflow as tf

from src.definitions import EMULATE_MODE
from src.utils.logger.logger_interface import ILogger
from src.model_trainer.mode_trainer_interface import IModeTrainer


class ModeTrainer(IModeTrainer):
    def __init__(self, logger: ILogger, ):
        self._logger = logger

    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int) -> tuple[
        tf.keras.Model, tf.keras.callbacks.History]:
        if EMULATE_MODE:
            time.sleep(0.5)  # simulate training process
            return model

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            restore_best_weights=True
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
        return model, history
