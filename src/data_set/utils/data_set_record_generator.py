import random
import json
import numpy as np

from src.data_set.utils.data_set_file_worker import DataSetFileWorker

from src.definitions import sr as SR, labels, DURATION

DEFAULT_DURATION = 2.0
count_of_fragments = int(DEFAULT_DURATION / DURATION)

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
        return timestamp, timestamp + duration

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

    def generate_test_record(self, duration=float, except_sets: list['str'] = [], except_labels: list[str] = []):
        between_records = []
        annotations = {}
        label_order = labels.copy()
        for label in except_labels:
            label_order.remove(label)
        label_order.remove('noise')
        label_records = [[] for _ in label_order]


        for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
            # transformations are only for train set
            if set_name in except_sets:
                continue
            if label in except_labels:
                continue
            if label == 'noise':
                between_records.append(signal)
                continue
            label_index = label_order.index(label)
            label_records[label_index].append(signal)
            annotations[label] = []

        test_record, indexes = self._get_random_records(between_records, count_of_fragments)
        # test_record = self.split_signal(test_record, DURATION)
        between_records = self._remove_by_indexes(between_records, indexes)
        timestamp = self._get_durations(0, test_record)[1]

        for index, records in enumerate(label_records):
            # get random record
            record, rec_rm_indexes = self._get_random_records(records, count_of_fragments)
            # get random between record
            between_record, bt_rm_indexes = self._get_random_records(between_records, count_of_fragments)
            # between_record = self.split_signal(between_record, duration)
            between_records = self._remove_by_indexes(between_records, bt_rm_indexes)

            # update current timestamp
            record_timestamp = self._get_durations(timestamp, record)
            timestamp = record_timestamp[1]
            # save annotation
            annotations[label_order[index]].append([record_timestamp[0], record_timestamp[1]])
            # update current timestamp
            between_record_timestamp = self._get_durations(timestamp, between_record)
            timestamp = between_record_timestamp[1]
            # merge records
            test_record = np.concatenate((test_record, record))
            test_record = np.concatenate((test_record, between_record))

        file_name = self.write_signal(test_record, SR, self.files.join(self.get_in_path(), 'records'), f'test')
        self._sve_annotation(annotation=annotations, path=file_name.replace('.wav', '.annotation.json'))
