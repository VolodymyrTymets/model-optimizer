import random

import tensorflow as tf

from src.definitions import EMULATE_MODE
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_interface import ILogger
from src.model_validator.mode_validator_interfcace import IModeValidator
from src.model_validator.model_record_evaluator.model_record_evaluator import ModelRecordEvaluator
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser


class ModeValidator(IModeValidator):
    def __init__(self, logger: ILogger, af_strategy: IAFStrategy):
        self._logger = logger
        self.model_record_evaluator = ModelRecordEvaluator(model_parser=ModelResultParser(af_strategy=af_strategy))

    def validate(self, model: tf.keras.Model, data: tf.data.Dataset, validation_records_path: str) -> tuple[
        float, float]:
        if EMULATE_MODE:
            record_acc, validation_acc = random.randint(1, 100), random.randint(1, 100)  # simulate validation process
            return record_acc, validation_acc
        evaluation = model.evaluate(data)
        test_acc, test_loss = evaluation
        validation_acc = self.model_record_evaluator.evaluate_records(model=model, from_path=validation_records_path)
        record_acc = test_acc
        return record_acc, validation_acc
