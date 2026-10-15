import json
import os
from typing import Sequence, Union

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.model_exporter.model_instance.model_instance import ModelInstance
from src.definitions import DURATION, sr

TFLITE_FILE_NAME = 'model.tflite'
SETTINGS_FILE_NAME = 'model_settings.json'


class ModelExporter:
    def __init__(self, af_strategy: IAFStrategy):
        self.files = Files()
        self.loger = Logger('ModelExporter')
        self.af_strategy = af_strategy

    def _get_export_path(self, path: str, subdir: str = ''):
        return self.files.join(path, subdir)

    def export_model(self, model, labels, path: str):
        export = ModelInstance(model, labels, self.af_strategy.AFType, str(DURATION))

        tf.saved_model.save(export, path, signatures={
            'get_settings': export.get_settings,
        })
        self.loger.log('Model is saved to: {}'.format(path), 'green')

    def export_tflite_model(self, model: tf.keras.Model, labels: Union[Sequence[str], np.ndarray], path: str) -> str:
        folder = self._get_export_path(path, 'tflite')
        self.files.create_folder(folder)
        tflite_path = self.files.join(folder, TFLITE_FILE_NAME)
        settings_path = self.files.join(folder, SETTINGS_FILE_NAME)
        tmp_tflite_path, tmp_settings_path = tflite_path + '.tmp', settings_path + '.tmp'

        try:
            # a fixed batch of 1 gives recurrent layers (GRU/LSTM) a static tensor-list shape, which the builtin
            # TFLite ops need; a dynamic batch fails with 'TensorListReserve requires element_shape to be static'
            static_input = tf.keras.Input(batch_shape=(1,) + tuple(model.input_shape[1:]), dtype=tf.float32)
            static_model = tf.keras.Model(static_input, model(static_input, training=False))
            tflite_model = tf.lite.TFLiteConverter.from_keras_model(static_model).convert()
            # the .tflite returns probabilities only, so the device reads names and input from this file
            settings = {
                'labels': [str(label) for label in labels],
                'af_type': self.af_strategy.AFType.value,
                'duration': str(DURATION),
                'sample_rate': sr,
                'input_shape': list(model.input_shape[1:]),
            }
            with open(tmp_tflite_path, 'wb') as tflite_file:
                tflite_file.write(tflite_model)
            with open(tmp_settings_path, 'w') as settings_file:
                json.dump(settings, settings_file)
            # renamed only when both are written, so the pair is never half there
            os.replace(tmp_tflite_path, tflite_path)
            os.replace(tmp_settings_path, settings_path)
        except Exception as error:
            self.loger.log(f'TFLite export failed for {folder}', 'red')
            self.loger.error(error)
            self.remove_tflite_export(path)
            raise

        self.loger.log('TFLite model is saved to: {}'.format(tflite_path), 'green')
        return tflite_path

    def remove_tflite_export(self, path: str) -> None:
        # a file left from a previous export would sit next to a new SavedModel and be shipped by mistake
        folder = self._get_export_path(path, 'tflite')
        for name in (TFLITE_FILE_NAME, SETTINGS_FILE_NAME, TFLITE_FILE_NAME + '.tmp', SETTINGS_FILE_NAME + '.tmp'):
            leftover = self.files.join(folder, name)
            try:
                if self.files.is_exist(leftover):
                    self.files.remove(leftover)
            except OSError as error:
                # keep going: the conversion error must stay the one that is raised
                self.loger.error(error)

    def load_model(self, path: str):
        self.loger.log(f'Loading model from: {path}', 'blue')
        return tf.saved_model.load(export_dir=path)

    def export_model_plot(self, model, path: str):
        self.files.create_folder(self._get_export_path(path))
        to_file = self.files.join(self._get_export_path(path), 'model_plot.png')
        tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True, show_layer_names=True,
                                  show_layer_activations=True,
                                  show_trainable=True)
        return to_file

    def export_training_plot(self, history: tf.keras.callbacks.History, path: str):
        to_file = self.files.join(self._get_export_path(path), 'training_plot.png')
        metrics, epoch = history.history, range(1, len(history.history['loss']) + 1)
        plt.rcParams.update({
            'font.size': 18,
        })
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')

        plt.subplot(1, 2, 2)
        plt.plot(epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')

        self.files.create_folder(self._get_export_path(path))
        plt.savefig(to_file)
        return to_file
