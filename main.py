from abc import ABC

from src.experiment.experiment_types import ExperimentDetails
from src.model_schema.model_schema_types import LayerType, OptimizerType, RegularizerType, LossType, ActivationType
from src.model_trainer.mode_trainer import ModeTrainer
from src.model_validator.mode_validator import ModeValidator
from src.model_builder.mode_builder import ModeBuilder
from src.experiment.experiment import Experiment
from src.utils.logger.logger_service import Logger
from src.database.db_client import DBClient


def main():
  # todo: add data-set import
  # todo: add SQLAlchemy to log the progress of the experiment
  # todo: implement models building
  # todo: implement models validation
  # todo: implement the experiment pipline
  db_client = DBClient()
  db_client.create_database()

  experiment = Experiment(
      logger=Logger('Experiment'),
      model_builder=ModeBuilder(logger=Logger('ModeBuilder')),
      mode_trainer=ModeTrainer(logger=Logger('ModeTrainer')),
      mode_validator=ModeValidator(logger=Logger('ModeValidator')),
      details=ExperimentDetails(
          epochs=100,
          batch_size=32,
          layers=[LayerType.Dense, LayerType.Conv, LayerType.GRU],
          activation=[ActivationType.ReLU, ActivationType.Sigmoid],
          units_range=[8, 128],
          optimizer=[OptimizerType.Adam, OptimizerType.AdamW],
          regularizer=[RegularizerType.L1, RegularizerType.L2],
          loss=[LossType.MSE, LossType.BinaryCrossentropy],
      )
  )

  experiment.run(test_data=None, train_data=None)

if __name__ == "__main__":
  main()