import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.definitions import FRAGMENT_LENGTH, sr
from src.utils.files import Files
from src.utils.logger.logger_service import Logger
from src.utils.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class DataSetImporter:
  def __init__(self, af_strategy: IAFStrategy, duration: float = 10.0):
    self.files = Files()
    self.loger = Logger('CNNModelPreprocessStrategy')
    self.af_strategy = af_strategy
    self.duration = duration
    self.shape = self._calculate_shape()

  def _get_fragment_length(self):
    return int(sr / (1 / self.duration))

  def _calculate_shape(self):
    signal = np.zeros(self._get_fragment_length())
    af = self.af_strategy.get_audio_feature(signal)
    return af.shape

  def get_bunch_audio_feature(self, waveform):
    bunch = waveform.numpy()
    return np.array([self.af_strategy.get_audio_feature(signal) for signal in bunch])

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def reshape(self, i):
    return tf.reshape(i, (-1,) + self.shape)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def get_audio_feature(self, i):
    return tf.py_function(self.get_bunch_audio_feature, [i], tf.float32)

  def import_data_set(self, data_set_path: str):
    # Form data storage
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_set_path, 'train'),
      batch_size=32,
      validation_split=0.2,
      seed=0,
      output_sequence_length=FRAGMENT_LENGTH,
      subset='both')

    test_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_set_path, 'test'),
      batch_size=32,
      seed=0,
      output_sequence_length=FRAGMENT_LENGTH)

    label_names = np.array(train_ds.class_names)

    # Prepare data - wave to audio feature
    train_ds = train_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)

    # Training model
    train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names
