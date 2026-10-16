from collections import Counter
from typing import Optional

import tensorflow as tf
import shutil
from src.data_set.types import ArgumentationTypes
from src.utils.files import Files


from src.data_set.utils.data_set_splitter import DataSetFileWorker
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.logger.logger_service import Logger
from src.assets_service.assets_service_interface import IAssetsService
from src.definitions import FRAGMENT_LENGTH, sr, frame_length, hop_length, keet_prefix
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy

FILTERED_MARKER = '__filtered__'


class LocalModelLoader:
    def __init__(self, af_strategy: IAFStrategy, assets_service: IAssetsService,
                 model_parser: ModelResultParser, model_name: str):
        self.assets_service = assets_service
        self.files = Files()
        self.logger = Logger('DataSetLocalFilter')
        self.model_parser = model_parser
        self.af_strategy = af_strategy
        self.model_name = model_name

    def _get_af_strategy(self, model, duration: float) -> Optional[IAFStrategy]:
        # a settings failure must stop the run: filtering with the wrong feature strategy would relabel files
        af_type, model_duration = self.model_parser.parse_settings(model=model)
        if model_duration != duration:
            raise ValueError(f'Model {self.model_name} was exported for duration {model_duration}, expected {duration}')
        if af_type.value != self.af_strategy.AFType.value:
            self.logger.log(f'Changing audio feature strategy to {af_type.value}', color='yellow')
            return AFStrategyFactory(sr=sr, frame_length=frame_length,
                                     hop_length=hop_length).create_strategy(strategy_type=af_type)
        return None

    def get_model(self, duration: float) -> tuple[tf.Module, Optional[IAFStrategy]]:
        models_path, _ = self.assets_service.get_models_path(self.model_name)
        model_path = self.files.join(models_path, self.model_name)
        if not self.files.is_exist(model_path):
            raise FileNotFoundError(f'Model folder not found: {model_path}')
        # the exporter writes a SavedModel with a get_settings signature; Keras 3 load_model rejects that format
        model = tf.saved_model.load(model_path)

        return model, self._get_af_strategy(model, duration)

    def is_filtered(self, db_path: str):
        return self.files.is_exist(self.files.join(db_path, FILTERED_MARKER))

    def finish(self, db_path: str):
        self.files.create_folder(self.files.join(db_path, FILTERED_MARKER))


class DataSetFilter(DataSetFileWorker):
    def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str],
                 assets_service: IAssetsService, af_strategy: IAFStrategy, model_name: str, dry_run: bool = False):
        super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
        self.assets_service = assets_service
        self.af_strategy = af_strategy
        self.model_parser = ModelResultParser(af_strategy=self.af_strategy)
        self.logger = Logger('DataSetFilter')
        self.except_sets = []
        self.except_labels = []
        self.dry_run = dry_run
        self.assets_service = assets_service
        self.model_loader = LocalModelLoader(af_strategy=self.af_strategy,
                                             assets_service=self.assets_service,
                                             model_parser=self.model_parser,
                                             model_name=model_name)

    def is_filtered(self):
        return self.model_loader.is_filtered(self.out_path)

    def finish(self):
        self.logger.log('Finishing', color='blue')
        self.model_loader.finish(self.out_path)

    def _check_folders_exist(self):
        # read_data_set lists every <set>/<label> folder; a missing one would fail midway, after files were moved
        for set_name in self.set_names:
            for label in self.labels:
                if not self.files.is_exist(self.files.join(self.in_path, set_name, str(label))):
                    raise FileNotFoundError(f'Data set folder missing for set "{set_name}", label "{label}"')

    def filter(self, duration: float) -> bool:
        self.logger.log('Start filtering', color='blue')
        if self.model_loader.is_filtered(self.out_path):
            self.logger.log('Filtering already done. Skipping.', color='blue')
            return False

        self._check_folders_exist()
        model, filter_af_strategy = self.model_loader.get_model(duration)
        if not model:
            self.logger.log('Model not found. Filtering skipped.', color='red')
            return False
        if filter_af_strategy is not None:
            self.model_parser = ModelResultParser(af_strategy=filter_af_strategy)
        moves = Counter()
        planned = []
        for signal, sr, set_name, label, path, file in self.read_data_set(log=False, exclude_labels=self.except_labels, except_sets=self.except_sets):
            if keet_prefix in file:
                continue
            if ArgumentationTypes.pitch_shift.value in file or ArgumentationTypes.time_shift.value in file or ArgumentationTypes.time_stretch.value in file or ArgumentationTypes.normalization.value in file:
                continue
            if len(signal) >= FRAGMENT_LENGTH:
                signal_label, _ = self.model_parser.parse(model=model, x=signal)
                if label != signal_label:
                    moves[(set_name, label, signal_label)] += 1
                    out_folder = self.files.join(self.out_path, set_name, signal_label)
                    planned.append((self.files.join(path, file), out_folder, self.files.join(out_folder, file),
                                    f'{set_name}/{label}/{file} -> {set_name}/{signal_label}'))

        # decide everything before moving anything: shutil.move overwrites an existing file silently
        targets = [to_path for _, _, to_path, _ in planned]
        clashes = len(targets) - len(set(targets)) + sum(1 for to_path in set(targets) if self.files.is_exist(to_path))
        if clashes:
            raise FileExistsError(f'{clashes} planned move(s) would overwrite an existing file; nothing was moved')
        is_filtered = len(planned) > 0
        verb = 'would move' if self.dry_run else 'moving'
        for *_, description in planned:
            self.logger.log(f'{verb} {description}', color='yellow')
        for (set_name, from_label, to_label), count in sorted(moves.items()):
            self.logger.log(f'{verb} {count} from {set_name}/{from_label} '
                            f'to {set_name}/{to_label}', color='yellow')
        if not self.dry_run:
            done = 0
            try:
                for from_path, out_folder, to_path, _ in planned:
                    self.files.create_folder(out_folder)
                    shutil.move(from_path, to_path)
                    done += 1
            except OSError:
                # no marker is written, so the half-filtered state must be visible in the log
                self.logger.log(f'Stopped after moving {done} of {len(planned)} files; the data set is partly filtered',
                                color='red')
                raise
        self.logger.log('End filtering', color='blue')
        if not self.dry_run:
            self.finish()
        return is_filtered
