from src.data_set.types import ArgumentationTypes
from src.experiment.experiment_types import ExperimentDetails
from src.experiments.experiments import Experiments
from src.model_schema.model_schema_types import LayerType, OptimizerType, RegularizerType, LossType, ActivationType
from src.utils.audio_features.types import AFTypes


def main():
    experiments = Experiments()
    experiments.run(
        af_types=[AFTypes.mfcc],
        argumentation_types = [ArgumentationTypes.time_shift, ArgumentationTypes.normalization, ArgumentationTypes.pitch_shift],
        model_setting=ExperimentDetails(
            epochs=100,
            batch_size=32,
            layers=[LayerType.Conv, LayerType.GRU, LayerType.Dense, ],
            activation=[ActivationType.ReLU, ActivationType.Sigmoid],
            units_range=[8, 256],
            optimizer=[OptimizerType.Adam, OptimizerType.AdamW],
            regularizer=[RegularizerType.L1, RegularizerType.L2],
            loss=[LossType.SparseCategoricalCrossentropy],
        )
    )

if __name__ == "__main__":
    main()
