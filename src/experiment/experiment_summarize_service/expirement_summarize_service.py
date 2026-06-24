import tensorflow as tf
import numpy as np
from os.path import join
from src.data_set.data_set_cooker import DataSetCooker
from src.experiment.experiment_summarize_service.expirement_summarize_service_interface import \
    IExperimentSummarizeService
from src.experiment.experiment_step.experiment_step import ExperimentStep
from src.experiment.experiment_types import IExperimentDetails
from src.experiment.models.experiment_model_service import ExperimentModelService
from src.experiment.models.experiment_step_model_service import ExperimentStepModelService
from src.model_builder.mode_builder import ModeBuilder
from src.model_trainer.mode_trainer import ModeTrainer
from src.model_validator.model_record_label.model_record_label import ModelRecordLabeler
from src.model_validator.model_record_evaluator.model_record_evaluator import ModelRecordEvaluator
from src.model_validator.mode_validator import ModeValidator
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.logger.logger_service import Logger
from src.model_exporter.model_exporter import ModelExporter


class ExperimentSummarizeService(IExperimentSummarizeService):
    def __init__(self, details: IExperimentDetails, af_strategy: IAFStrategy):
        self.logger = Logger('ExperimentSummarizeService')
        self._experiment_model_service = ExperimentModelService(Logger('ExperimentModelService'))

        self._experiment_step_model_service = ExperimentStepModelService(Logger('ExperimentStepModelService'))
        self.mode_builder = ModeBuilder(logger=Logger('ModeBuilder'))
        self.mode_trainer = ModeTrainer(logger=Logger('ModeTrainer'))
        self.mode_validator = ModeValidator(logger=Logger('ModeValidator'), af_strategy=af_strategy)
        self.model_record_evaluator = ModelRecordEvaluator(ModelResultParser(af_strategy=af_strategy))

        self._experiment_model = self._experiment_model_service.get_current_experiment(details)
        self._details = self._experiment_model_service.get_details(self._experiment_model)
        self.data_set_cooker = DataSetCooker(experiment_id=self._experiment_model.id)
        self.result_path = join(self.data_set_cooker.get_experiment_path(), 'results')
        self.model_record_label_service = ModelRecordLabeler(ModelResultParser(af_strategy=af_strategy), export_path=self.result_path)
        self._experiment_step = ExperimentStep(self._experiment_model.id, af_strategy)
        self.model_exporter = ModelExporter()

    def _log_experiment_details(self, best_step, best_schema, record_acc, validation_acc):
        self.logger.log(f"Experiment {self._experiment_model.id} summary:")
        self.logger.log("")
        self.logger.log(
            f"Best step [{best_step.record_accuracy}, {best_step.validation_accuracy}] - {best_step.step}, with id {best_step.id}")
        self.logger.log(f"schema: {str(best_schema)}")
        self.logger.log(f"Record accuracy: {record_acc}, Validation accuracy: {validation_acc}")
        self.logger.log("")

    def summarize(self, data_sets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], labels: list[str]) -> None:
        self.logger.log("Experiment summarized started", color="green")
        train_ds, val_ds, test_ds = data_sets
        best_step = self._experiment_step_model_service.get_best_step(self._experiment_model.id)
        best_schema = self._experiment_step.get_schema(step=best_step)
        model = self.mode_builder.build_model(best_schema, train_ds)
        model, history = self.mode_trainer.train(model, train_ds, val_ds, self._details.epochs)
        record_acc, validation_acc = self.mode_validator.validate(model=model, data=test_ds, validation_records_path=self.data_set_cooker.get_validation_records_path())
        self.model_record_label_service.label_records(model=model, from_path=self.data_set_cooker.get_validation_records_path())
        self.model_exporter.export_model(model, path=self.data_set_cooker.get_model_path(), labels=labels)
        self.model_exporter.export_model_plot(model, path=self.data_set_cooker.get_model_path())
        self.model_exporter.export_training_plot(history, path=self.data_set_cooker.get_model_path())
        self.logger.log("Experiment summarized finished", color="green")

        self._log_experiment_details(best_step, best_schema, record_acc, validation_acc)
