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


class Experiments:
    def __init__(self):
        self.db_client = DBClient()
        self.db_client.create_database()

    def create_experiment(self, experiment_details: IExperimentDetails, data_set_details: IExperimentDataSetDetails,
                          af_strategy: IAFStrategy):
        return Experiment(
            details=experiment_details,
            data_set_details=data_set_details,
            af_strategy=af_strategy
        )

    def prepare_data_set(self, experiment: Experiment, af_strategy: IAFStrategy,
                         argumentation_types: list[ArgumentationTypes]):
        data_set_cooker = DataSetCooker(experiment_id=experiment.get_experiment_id(), af_strategy=af_strategy)
        data_set_cooker.prepare(duration=DURATION, argumentation_types=argumentation_types)

        data_set_importer = DataSetImporter(experiment_id=experiment.get_experiment_id(), duration=DURATION,
                                            af_strategy=af_strategy)
        train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()
        return train_ds, val_ds, test_ds, label_names

    def train_experiment(self, experiment: Experiment, af_strategy: IAFStrategy):


        experiment.start()
        data_set_importer = DataSetImporter(experiment_id=experiment.get_experiment_id(), duration=DURATION,
                                            af_strategy=af_strategy)
        train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()
        experiment.summarize(data_sets=(train_ds, val_ds, test_ds), labels=label_names)
        # raise Exception("!!! STOP")
        experiment.finish()

    def run(self, af_types: list[AFTypes], argumentation_types: list[ArgumentationTypes],
            model_setting: IExperimentDetails, train: bool = True):
        exp_argumentation_types = []
        for af_type in af_types:
            for argumentation_type in argumentation_types:
                exp_argumentation_types.append(argumentation_type)
                data_set_details = ExperimentDataSetDetails(
                    duration=DURATION, labels=labels, argumentation_types=exp_argumentation_types,
                    af_type=af_type
                )
                af_strategy = AFStrategyFactory(sr=sr, frame_length=frame_length,
                                                hop_length=hop_length).create_strategy(af_type)
                experiment = self.create_experiment(experiment_details=model_setting, data_set_details=data_set_details,
                                                    af_strategy=af_strategy)
                if experiment.is_finished():
                    continue
                self.prepare_data_set(experiment=experiment,
                                      af_strategy=af_strategy,
                                      argumentation_types=exp_argumentation_types)
                if train:
                    self.train_experiment(experiment=experiment, af_strategy=af_strategy)
            exp_argumentation_types = []
