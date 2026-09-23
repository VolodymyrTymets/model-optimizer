import numpy as np
import tensorflow as tf

from src.definitions import ASSETS_PATH, DATA_SET_NAME, IMAGE_SIZE, IMAGE_COLOR_MODE
from src.utils.files import Files
from src.utils.logger.logger_service import Logger


class ImageDataSetImporter:
  def __init__(self, batch_size: int = 32, seed: int = 0):
    self.files = Files()
    self.loger = Logger('ImageDataSetImporter')
    self.batch_size = batch_size
    self.seed = seed

  def import_data_set(self):
    data_set_path = self.files.join(ASSETS_PATH, DATA_SET_NAME)
    self.loger.log(f"Importing image data set from {data_set_path}", color="green")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      directory=self.files.join(data_set_path, 'train'),
      color_mode=IMAGE_COLOR_MODE,
      image_size=IMAGE_SIZE,
      batch_size=self.batch_size,
      validation_split=0.2,
      seed=self.seed,
      subset='both')

    test_ds = tf.keras.utils.image_dataset_from_directory(
      directory=self.files.join(data_set_path, 'test'),
      color_mode=IMAGE_COLOR_MODE,
      image_size=IMAGE_SIZE,
      batch_size=self.batch_size,
      seed=self.seed)

    label_names = np.array(train_ds.class_names)

    train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names
