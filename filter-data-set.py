import argparse
import os

from src.assets_service.assets_service import AssetsService
from src.data_set.utils.data_set_filter import DataSetFilter
from src.definitions import DURATION, frame_length, hop_length, labels, sr, sub_sets
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.audio_features.types import AFTypes


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Move mislabeled fragments of the cooked data set to the label an exported model predicts.')
    parser.add_argument('model_name', help='folder under assets/models, e.g. model-breath_noise_stimulation-0.5-mfcc')
    parser.add_argument('--dry-run', action='store_true', help='log what would move, move nothing')
    args = parser.parse_args()

    assets_service = AssetsService(experiment_id=None)
    # raises ValueError on a name that is not model-<labels>-<duration>-<af_type>
    labels_part, duration_part, af_part = assets_service.parse_model_name(args.model_name)
    data_set_path = assets_service.get_data_set_path_for_model(args.model_name)

    # the name carries no task, so a model from another task is caught here by its labels and duration
    expected_labels = '_'.join(sorted(labels))
    if labels_part != expected_labels or duration_part != str(DURATION):
        raise ValueError(f'Model {args.model_name} is for labels "{labels_part}" and duration {duration_part}, '
                         f'but the active task has labels "{expected_labels}" and duration {DURATION}')
    try:
        af_type = AFTypes[af_part]
    except KeyError:
        raise ValueError(f'Unknown audio feature type "{af_part}" in model name {args.model_name}') from None

    if not os.path.isdir(data_set_path):
        raise FileNotFoundError('Cooked data set not found for this model; run main.py once first to create it')

    data_set_filter = DataSetFilter(
        in_path=data_set_path, out_path=data_set_path, sub_sets=sub_sets, labels=labels,
        assets_service=assets_service,
        af_strategy=AFStrategyFactory(sr=sr, frame_length=frame_length, hop_length=hop_length).create_strategy(af_type),
        model_name=args.model_name, dry_run=args.dry_run)
    if not data_set_filter.is_filtered() and not args.dry_run:
        print('Filtering the cooked data set in place; this script cannot move files back.')
    is_filtered = data_set_filter.filter(duration=DURATION)
    print(f'is_filtered: {is_filtered}')


if __name__ == "__main__":
    main()
