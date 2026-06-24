import os
import numpy as np
import json
import tensorflow as tf

from src.model_validator.model_result_parser.model_result_parser_interaace import IModelResultParser
from src.definitions import FRAGMENT_LENGTH, labels
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.utils.wav_files import WavFiles
from src.utils.lists import to_chunks

class ModelRecordEvaluator:
    def __init__(self, model_parser: IModelResultParser ):
        self.files = Files()
        self.wav_files = WavFiles()
        self.model_parser = model_parser
        self.loger = Logger('ModelRecordEvaluator')

    def _load_annotation(self, file_pat: str):
        annotation_path = file_pat.replace('.wav', f'.annotation.json')
        annotations = {}
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
        return annotations

    def is_in_timestamp(self, start: float, end: float, timestamps: list[list[float]]):
        is_in = False
        for timestamp in timestamps:
            if start >= timestamp[0] and end <= timestamp[1]:
                is_in = True
                break
        return is_in

    def evaluate_record(self, model: tf.keras.Model, file_path: str):
        waveform, sr = self.wav_files.read(file_path)
        annotations = self._load_annotation(file_path)
        if not annotations.keys():
            return 0
        model_annotation = []

        timestamp = 0
        chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
        rate_per_chunk = 100 / len(chunks)
        evaluate_rate = 0

        for chunk in chunks:
            duration = 1 / sr * len(chunk)
            start = timestamp
            end = timestamp + duration
            line_label, prediction = self.model_parser.parse(model, chunk)
            model_annotation.append([start, end, line_label])
            annotation_label = labels[0]
            for key in annotations.keys():
                timestamps = annotations[key]
                if self.is_in_timestamp(start, end, timestamps):
                    annotation_label = key
                    break
            chunk_evaluate_rate = rate_per_chunk if line_label == annotation_label else 0
            timestamp += duration
            evaluate_rate += chunk_evaluate_rate
        self.loger.log(f'Record labels: {file_path}:')
        self.loger.log(f'Evaluate rate: {evaluate_rate} %')
        return evaluate_rate

    def evaluate_records(self, model: tf.keras.Model, from_path: str):
        test_record_acc = []
        for file in self.files.get_only_files(from_path):
            if file.endswith('.wav'):
                record_acc = self.evaluate_record(model, self.files.join(from_path, file))
                test_record_acc.append(record_acc)
        mean_acc = np.mean(test_record_acc)
        self.loger.log(f'Mean records accuracy: {mean_acc}%', color='blue')
        return float(mean_acc)
