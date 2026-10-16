import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data_set.utils import data_set_record_generator as generator_module
from src.data_set.utils.data_set_record_generator import DataSetRecordGenerator

SR = generator_module.SR
FRAGMENT_DURATION = 0.5
FRAGMENT_LENGTH = int(FRAGMENT_DURATION * SR)
# amplitudes tell the fragments apart in the written record (PCM_16 keeps them well apart)
KEEP_LEVEL = 0.6
NORMAL_LEVEL = 0.1
TRAIN_KEEP_LEVEL = 0.9
LABELS = ['noise', 'stimulation', 'breath']
# 4 fragments per segment (2.0 s / 0.5 s) and 5 segments (noise, then label + noise for two labels)
SLOTS_PER_RECORD = 20
MIN_KEEP_PER_RECORD = 6


@pytest.fixture(autouse=True)
def seeded(monkeypatch):
    monkeypatch.setattr(generator_module, 'labels', LABELS)
    random.seed(0)
    np.random.seed(0)


def _write_fragment(path: Path, level: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full(FRAGMENT_LENGTH, level, dtype=np.float32), SR)


@pytest.fixture
def make_data_set(tmp_path):
    def make(keep_per_label: dict, normal_per_label: int = 40) -> DataSetRecordGenerator:
        in_path = tmp_path / 'data_set'
        for label in LABELS:
            for i in range(normal_per_label):
                _write_fragment(in_path / 'test' / label / f'{i}.wav', NORMAL_LEVEL)
            for i in range(keep_per_label.get(label, 0)):
                _write_fragment(in_path / 'test' / label / f'{i}_keep_{i}.wav', KEEP_LEVEL)
            # records are built from the test set only
            _write_fragment(in_path / 'train' / label / '0_keep_0.wav', TRAIN_KEEP_LEVEL)
        out_path = tmp_path / 'records'
        out_path.mkdir()
        return DataSetRecordGenerator(in_path=str(in_path), out_path=str(out_path), sub_sets=['train', 'test'],
                                      labels=LABELS)

    return make


def _generate(generator: DataSetRecordGenerator, record_count: int) -> list:
    for _ in range(record_count):
        generator.generate_test_record(duration=FRAGMENT_DURATION, except_sets=['train'], except_labels=[])
    records = []
    for wav in sorted(Path(generator.out_path).glob('*.wav')):
        signal, _ = sf.read(str(wav))
        records.append(signal.reshape(-1, FRAGMENT_LENGTH).mean(axis=1))
    return records


def _keep_counts(records: list) -> list:
    return [int((fragments > (KEEP_LEVEL + NORMAL_LEVEL) / 2).sum()) for fragments in records]


# keep pools are tiny against 40 normal fragments per label, so random picks alone give ~0-1 keep fragments per record
@pytest.mark.parametrize('record_count', [1, 5], ids=['one-record', 'five-records'])
@pytest.mark.parametrize('keep_per_label', [
    {'noise': 2, 'stimulation': 2, 'breath': 2},
    {'noise': 1, 'stimulation': 1, 'breath': 1},
    {'noise': 1},
], ids=['two-per-label', 'one-per-label', 'noise-only'])
def test_every_record_has_at_least_30_percent_keep_flagged_fragments(make_data_set, keep_per_label, record_count):
    records = _generate(make_data_set(keep_per_label), record_count)

    assert [len(fragments) for fragments in records] == [SLOTS_PER_RECORD] * record_count
    assert min(_keep_counts(records)) >= MIN_KEEP_PER_RECORD


def test_a_single_keep_flagged_fragment_fills_the_quota_by_repeating_itself(make_data_set, capsys):
    # documents current behaviour (picks are with replacement): the quota counts slots, not distinct fragments
    records = _generate(make_data_set({'noise': 1}), 1)

    assert min(_keep_counts(records)) >= MIN_KEEP_PER_RECORD
    assert 'Not enough' not in capsys.readouterr().out


def test_every_record_includes_a_keep_flagged_fragment(make_data_set):
    records = _generate(make_data_set({'stimulation': 1}), 5)

    assert min(_keep_counts(records)) >= 1


def test_keep_flagged_fragments_of_the_train_set_are_ignored(make_data_set):
    records = _generate(make_data_set({'noise': 3}), 3)

    assert max(fragments.max() for fragments in records) < TRAIN_KEEP_LEVEL - 0.1


def test_no_keep_flagged_fragment_still_generates_records_and_warns(make_data_set, capsys):
    records = _generate(make_data_set({}), 2)

    assert len(records) == 2
    assert _keep_counts(records) == [0, 0]
    assert f'0 of {MIN_KEEP_PER_RECORD}' in capsys.readouterr().out


def test_quota_is_capped_by_what_the_labels_can_supply_and_warns(make_data_set, capsys):
    # one label segment holds 4 slots, so 4 is all the keep fragments that can be placed
    records = _generate(make_data_set({'stimulation': 1}), 2)

    assert _keep_counts(records) == [4, 4]
    assert f'4 of {MIN_KEEP_PER_RECORD}' in capsys.readouterr().out


@pytest.mark.parametrize('capacities, quota, expected', [
    ([4, 4, 4, 4, 4], 6, [2, 1, 1, 1, 1]),
    ([4, 0, 4, 0, 4], 6, [2, 0, 2, 0, 2]),
    ([4, 0, 0], 6, [4, 0, 0]),
    ([4, 4], 0, [0, 0]),
], ids=['spread', 'skips-empty-pools', 'capped', 'no-quota'])
def test_allocate_keep_slots_spreads_round_robin_and_caps(capacities, quota, expected):
    generator = DataSetRecordGenerator(in_path='in', out_path='out', sub_sets=['test'], labels=LABELS)

    assert generator._allocate_keep_slots(capacities, quota) == expected
