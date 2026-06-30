from typing import Any, Generator

import tensorflow as tf
from google.protobuf import duration
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from src.model_validator.model_record_label.model_record_label_interface import IModelRecordLabeler
from src.model_validator.model_result_parser.model_result_parser_interface import IModelResultParser
from src.definitions import FRAGMENT_LENGTH, labels, labels_colors
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.utils.wav_files import WavFiles
from src.utils.lists import to_chunks


class ModelRecordLabeler(IModelRecordLabeler):
    def __init__(self, model_parser: IModelResultParser, export_path: str = ''):
        self.files = Files()
        self.wav_files = WavFiles()
        self.model_parser = model_parser
        self.loger = Logger('ModelRecordLabeler')
        self.export_path = export_path

    def _get_color(self, label: str, alpha:float=0.5):
        color = labels_colors[label]
        return mcolors.to_rgba(color, alpha=alpha)

    def _add_legend(self, ax, all_labels, segments_labels):
        legends = []
        legend_labels = []
        for label in all_labels:
            if label in segments_labels:
                legends.append(Line2D([0], [0], color=self._get_color(label), label=label))
                legend_labels.append(str(label).capitalize())
        ax.legend(legends, legend_labels, loc='upper right')

    def _save_plot(self, file_name: str, segments, colors, segments_labels) -> str:
        plt.rcParams.update({
            'font.size': 10,
            'legend.fontsize': 14,
        })
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.add_collection(LineCollection(segments=segments, colors=colors))
        # Set x and y limits... sadly this is not done automatically for line
        ax.set_xlim(0, len(segments[0]) * len(segments))
        ax.set_ylim(1, -1)

        self._add_legend(ax, labels, segments_labels)

        self.files.create_folder(self.export_path)
        fig_path = self.files.join(self.export_path, file_name.replace('.wav', '.png'))
        plt.savefig(fig_path)
        self.loger.log(f'-> {fig_path} is saved', color='green')
        return fig_path

    def label_record(self, model: tf.keras.Model, file_path: str, file_name: str = ''):
        waveform, _ = self.wav_files.read(file_path)

        segments = []
        colors = []
        line_labels = []
        x = 0
        for i, chunk in enumerate(to_chunks(waveform, int(FRAGMENT_LENGTH))):
            segment = []
            for y in chunk:
                segment.append((x, y))
                x = x + 1
            segments.append(segment)
            line_label, prediction = self.model_parser.parse(model, chunk)
            color = self._get_color(line_label, alpha=prediction)
            colors.append(color)
            line_labels.append(line_label)

        return self._save_plot(file_name, segments, colors, line_labels)

    def label_records(self, model: tf.keras.Model, from_path: str) -> Generator[str, Any, None]:
        for file in self.files.get_only_files(from_path):
            if file.endswith('.wav'):
                yield self.label_record(model, self.files.join(from_path, file), file)
