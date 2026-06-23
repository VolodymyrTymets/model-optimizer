from src.data_set.types import ArgumentationTypes
from src.experiment.experiment_types import ExperimentDetails
from src.model_schema.model_schema_types import LayerType, OptimizerType, RegularizerType, LossType, ActivationType
from src.experiment.experiment import Experiment
from src.data_set.data_set import DataSet

from src.database.db_client import DBClient


def main():
    # todo: add data-set import
    # todo: implement models building
    # todo: implement models validation
    db_client = DBClient()
    db_client.create_database()

    experiment = Experiment(
        details=ExperimentDetails(
            epochs=100,
            batch_size=32,
            layers=[LayerType.Conv, LayerType.GRU, LayerType.Dense, ],
            activation=[ActivationType.ReLU, ActivationType.Sigmoid],
            units_range=[8, 256],
            optimizer=[OptimizerType.Adam, OptimizerType.AdamW],
            regularizer=[RegularizerType.L1, RegularizerType.L2],
            loss=[LossType.MSE, LossType.BinaryCrossentropy],
        )
    )
    data_set = DataSet(experiment_id=experiment.get_experiment_id())
    data_set.prepare(duration=0.5, argumentation_types=[ArgumentationTypes.normalization])

    # experiment.start(test_data=None, train_data=None)
    # experiment.finish()


if __name__ == "__main__":
    main()
