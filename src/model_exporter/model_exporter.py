import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.model_exporter.model_instance.model_instance import ModelInstance
from src.definitions import DURATION


class ModelExporter:
    def __init__(self, af_strategy: IAFStrategy):
        self.files = Files()
        self.loger = Logger('ModelExporter')
        self.af_strategy = af_strategy
        self._signature = f'_{DURATION}_{str(self.af_strategy.AFType.value)}'

    def _get_export_path(self, path: str):
        return path + self._signature

    def export_model(self, model, labels, path: str):
        export = ModelInstance(model, labels, self.af_strategy.AFType, str(DURATION))

        tf.saved_model.save(export, self._get_export_path(path), signatures={
            'get_settings': export.get_settings,
        })
        self.loger.log('Model is saved to: {}'.format(path), 'green')

    def load_model(self, path: str):
        self.loger.log(f'Loading model from: {path}', 'blue')
        return tf.saved_model.load(export_dir=path)

    def export_model_plot(self, model, path: str):
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
