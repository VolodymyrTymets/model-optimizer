import math
import random
import json
from typing import List

import numpy as np

from src.data_set.utils.data_set_file_worker import DataSetFileWorker

from src.definitions import sr as SR, labels, keet_prefix, KEEP_RECORDS_SHARE

DEFAULT_DURATION = 2.0

class DataSetRecordGenerator(DataSetFileWorker):
    def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
        super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)

    def _get_random(self, list_records: list[list[int]]):
        index = random.choice(range(len(list_records)))
        picked = list_records[index].copy()
        return picked, index

    def _remove_by_index(self, list_records: list[list[int]], index: int):
        if len(list_records) > 2:
            return np.delete(list_records, index, axis=0)
        return list_records

    def split_signal(self, signal: np.ndarray, duration: float):
        return signal[:int(duration * SR)]

    def _get_durations(self, timestamp: float, signal: np.ndarray):
        duration = 1 / SR * len(signal)
        return round(timestamp), round(timestamp + duration)

    def _sve_annotation(self, annotation: dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

    def _get_random_records(self, list_records: list[list[int]], count: int):
        records = []
        indexes = []
        for _ in range(count):
            test_record, index = self._get_random(list_records)
            records.append(test_record)
            indexes.append(index)
        return np.concatenate(records), indexes

    def _remove_by_indexes(self, list_records: list[list[int]], indexes: list[int]):
        for index in indexes:
            self._remove_by_index(list_records, index)
        return list_records

    def _allocate_keep_slots(self, capacities: List[int], quota: int) -> List[int]:
        # one slot at a time so the keep fragments spread over the record instead of filling its first segments
        allocated = [0] * len(capacities)
        remaining = min(quota, sum(capacities))
        while remaining > 0:
            for index, capacity in enumerate(capacities):
                if remaining > 0 and allocated[index] < capacity:
                    allocated[index] += 1
                    remaining -= 1
        return allocated

    def _pick(self, list_records: list, count: int):
        return [self._get_random(list_records)[0] for _ in range(count)]

    def _get_segment(self, records: list, keep_records: list, count: int, keep_count: int):
        fragments = self._pick(keep_records, keep_count) + self._pick(records, count - keep_count)
        random.shuffle(fragments)
        return np.concatenate(fragments)

    def generate_test_record(self, duration: float, except_sets: list['str'] = [], except_labels: list[str] = [], between_fragment_label = 'noise', keep_share: float = KEEP_RECORDS_SHARE):
        count_of_fragments = int(DEFAULT_DURATION / duration)
        annotations = {
            between_fragment_label: [],
        }
        label_order = labels.copy()
        for label in except_labels:
            label_order.remove(label)
        try:
            label_order.remove(between_fragment_label)
        except ValueError:
            pass
        label_records = [[] for _ in label_order]
        keep_label_records = [[] for _ in label_order]
        between_records = []
        keep_between_records = []

        for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
            # transformations are only for train set
            if set_name in except_sets:
                continue
            if label in except_labels:
                continue
            is_keep = keet_prefix in file
            if label == between_fragment_label:
                between_records.append(signal)
                if is_keep:
                    keep_between_records.append(signal)
                continue
            label_index = label_order.index(label)
            label_records[label_index].append(signal)
            if is_keep:
                keep_label_records[label_index].append(signal)
            annotations[label] = []

        # segments of one record in order: noise, then (label, noise) per label; a keep fragment may only fill a
        # slot of the pool it belongs to
        keep_pools = [keep_between_records]
        for keep_records in keep_label_records:
            keep_pools += [keep_records, keep_between_records]
        quota = math.ceil(keep_share * count_of_fragments * len(keep_pools))
        keep_counts = self._allocate_keep_slots([count_of_fragments if pool else 0 for pool in keep_pools], quota)
        if sum(keep_counts) < quota:
            self.logger.log(f'Not enough keep-flagged fragments for a record: {sum(keep_counts)} of {quota}', color='yellow')
        keep_counts = iter(keep_counts)

        # first fragment we took from between_records
        test_record = self._get_segment(between_records, keep_between_records, count_of_fragments, next(keep_counts))
        timestamp = self._get_durations(0, test_record)[1]
        # save annotation for between record
        annotations[between_fragment_label].append([0, timestamp])

        for index, records in enumerate(label_records):
            # get random record
            record = self._get_segment(records, keep_label_records[index], count_of_fragments, next(keep_counts))
            # get random between record
            between_record = self._get_segment(between_records, keep_between_records, count_of_fragments, next(keep_counts))

            # update current timestamp
            record_timestamp = self._get_durations(timestamp, record)
            timestamp = record_timestamp[1]
            # save annotation
            annotations[label_order[index]].append([record_timestamp[0], record_timestamp[1]])
            # update current timestamp
            between_record_timestamp = self._get_durations(timestamp, between_record)
            timestamp = between_record_timestamp[1]
            # save annotation for between record
            annotations[between_fragment_label].append([between_record_timestamp[0], between_record_timestamp[1]])
            # merge records
            test_record = np.concatenate((test_record, record))
            test_record = np.concatenate((test_record, between_record))

        file_name = self.write_signal(test_record, SR, self.out_path, f'test')
        self._sve_annotation(annotation=annotations, path=file_name.replace('.wav', '.annotation.json'))
