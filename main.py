from src.data_set.types import ArgumentationTypes
from src.definitions import DATA_SET_TYPE
from src.experiment.experiment_types import ExperimentDetails
from src.experiments.experiments import Experiments
from src.model_schema.model_schema_types import LayerType, OptimizerType, RegularizerType, LossType, ActivationType
from src.utils.audio_features.types import AFTypes


def main():
    experiments = Experiments()
    if DATA_SET_TYPE == 'image':
        experiments.run(
            argumentation_types=[ArgumentationTypes.nothing],
            model_setting=ExperimentDetails(
                epochs=100,
                batch_size=32,
                layers=[LayerType.Conv, LayerType.Conv, LayerType.Conv, LayerType.Dense],
                activation=[ActivationType.ReLU],
                units_range=[16, 128],
                optimizer=[OptimizerType.Adam],
                regularizer=[RegularizerType.L1],
                loss=[LossType.SparseCategoricalCrossentropy],
            ),
            train=True,
        )
        return
    experiments.run(
        af_types=[AFTypes.mfcc],
        argumentation_types = [ArgumentationTypes.nothing, ArgumentationTypes.time_shift, ArgumentationTypes.pitch_shift, ArgumentationTypes.normalization],
        model_setting=ExperimentDetails(
            epochs=100,
            batch_size=32,
            layers=[LayerType.Conv, LayerType.GRU, LayerType.Dense],
            activation=[ActivationType.ReLU, ActivationType.Sigmoid],
            units_range=[8, 1024],
            optimizer=[OptimizerType.Adam, OptimizerType.AdamW],
            regularizer=[RegularizerType.L1, RegularizerType.L2],
            loss=[LossType.SparseCategoricalCrossentropy],
        ),
        train=True,
    )

if __name__ == "__main__":
    main()
