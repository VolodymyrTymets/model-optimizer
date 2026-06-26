import time
from typing import Any, Union

import tensorflow as tf
from keras import Model
from src.definitions import EMULATE_MODE
from src.utils.logger.logger_interface import ILogger
from src.model_trainer.mode_trainer_interface import IModeTrainer

class MockHistory:
    def __init__(self, history: dict[str, list[int]]):
        self.history = history


class ModeTrainer(IModeTrainer):
    def __init__(self, logger: ILogger, ):
        self._logger = logger
        self._mock_history = {
            'loss': [0],
            'val_loss': [0],
            'accuracy': [0],
            'val_accuracy': [0],
        }

    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int) -> Union[
        tuple[Model, dict[str, dict[str, Union[list[int], int]]]], tuple[Model, Any]]:
        if EMULATE_MODE:
            time.sleep(0.5)  # simulate training process
            return model, MockHistory(self._mock_history)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            restore_best_weights=True
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
        return model, history
