from src.data_set.types import ArgumentationTypes
from src.experiment.experiment_types import ExperimentDetails
from src.model_schema.model_schema_types import LayerType, OptimizerType, RegularizerType, LossType, ActivationType
from src.experiment.experiment import Experiment
from src.data_set.data_set_cooker import DataSetCooker
from src.data_set.data_set_importer import DataSetImporter

from src.database.db_client import DBClient
from src.definitions import DURATION, sr, frame_length, hop_length, af_type
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory


def main():
    # todo: add data-set import
    # todo: implement models building
    # todo: implement models validation
    db_client = DBClient()
    db_client.create_database()

    af_strategy = AFStrategyFactory(sr=sr, frame_length=frame_length, hop_length=hop_length).create_strategy(af_type)

    experiment = Experiment(
        details=ExperimentDetails(
            epochs=100,
            batch_size=32,
            layers=[LayerType.Conv, LayerType.GRU, LayerType.Dense, ],
            activation=[ActivationType.ReLU, ActivationType.Sigmoid],
            units_range=[8, 256],
            optimizer=[OptimizerType.Adam, OptimizerType.AdamW],
            regularizer=[RegularizerType.L1, RegularizerType.L2],
            loss=[LossType.SparseCategoricalCrossentropy],
        ),
        af_strategy=af_strategy
    )
    data_set_cooker = DataSetCooker(experiment_id=experiment.get_experiment_id())
    data_set_cooker.prepare(duration=DURATION, argumentation_types=[])

    data_set_importer = DataSetImporter(duration=DURATION, af_strategy=af_strategy)
    train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set(data_set_cooker.get_data_set_path())

    experiment.start((train_ds, val_ds, test_ds))
    experiment.finish()


if __name__ == "__main__":
    main()
