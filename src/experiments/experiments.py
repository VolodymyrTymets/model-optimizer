from src.data_set.data_set_cooker import DataSetCooker
from src.data_set.types import ArgumentationTypes
from src.definitions import sr, frame_length, hop_length, labels
from src.experiment.experiment import Experiment
from src.experiment.experiment_types import IExperimentDetails, IExperimentDataSetDetails, ExperimentDataSetDetails
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.audio_features.types import AFTypes
from src.data_set.data_set_importer import DataSetImporter
from src.database.db_client import DBClient
from src.definitions import DURATION


class Experiments():
    def __init__(self):
        self.db_client = DBClient()
        self.db_client.create_database()

    def run_experiment(self, experiment_details: IExperimentDetails, data_set_details: IExperimentDataSetDetails,
                       af_strategy: IAFStrategy):
        experiment = Experiment(
            details=experiment_details,
            data_set_details=data_set_details,
            af_strategy=af_strategy
        )
        data_set_cooker = DataSetCooker(experiment_id=experiment.get_experiment_id(), af_strategy=af_strategy)
        data_set_cooker.prepare(duration=DURATION, argumentation_types=[])

        data_set_importer = DataSetImporter(experiment_id=experiment.get_experiment_id(), duration=DURATION,
                                            af_strategy=af_strategy)
        train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()

        experiment.start((train_ds, val_ds, test_ds))
        experiment.summarize((train_ds, val_ds, test_ds), labels=label_names)

        experiment.finish()

    def run(self, af_types: list[AFTypes], argumentation_types: list[ArgumentationTypes],
            model_setting: IExperimentDetails):
        exp_argumentation_types = []
        for af_type in af_types:
            for argumentation_type in argumentation_types:
                exp_argumentation_types.append(argumentation_type)
                af_strategy = AFStrategyFactory(sr=sr, frame_length=frame_length,
                                                hop_length=hop_length).create_strategy(af_type)
                self.run_experiment(experiment_details=model_setting, data_set_details=ExperimentDataSetDetails(
                    duration=DURATION, labels=labels, argumentation_types=exp_argumentation_types,
                    af_type=af_type
                ), af_strategy=af_strategy)
