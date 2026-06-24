import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.model_exporter.model_instance.model_instance import ModelInstance


class ModelExporter:
    def __init__(self):
        self.files = Files()
        self.loger = Logger('ModelExporter')

    def export_model_plot(self, model, path: str):
        tf.keras.utils.plot_model(model, to_file=
        self.files.join(path, 'model_plot.png'), show_shapes=True, show_layer_names=True, show_layer_activations=True,
                                  show_trainable=True)

    def export_model(self, model, labels, path: str):
        export = ModelInstance(model, labels)
        tf.saved_model.save(export, path)
        self.loger.log('Model is saved to: {}'.format(path), 'green')

    def load_model(self, path: str):
        self.loger.log(f'Loading model from: {path}', 'blue')
        return tf.saved_model.load(export_dir=path)

    def export_training_plot(self, history: tf.keras.callbacks.History, path: str):
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

        self.files.create_folder(path)
        plt.savefig(self.files.join(path, 'training_history.png'))
