import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models

from src.definitions import ASSETS_PATH, DATA_SET_NAME

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 0
input_shape = IMAGE_SIZE + (1,)

data_set_path = os.path.join(ASSETS_PATH, DATA_SET_NAME)
output_path = os.path.join(ASSETS_PATH, 'rare-model-image')


def import_data_set():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_set_path, 'train'),
        color_mode='grayscale',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=SEED,
        subset='both')

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_set_path, 'test'),
        color_mode='grayscale',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED)

    label_names = np.array(train_ds.class_names)

    train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names


def build_model(label_names):
    model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_names), activation='softmax')
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return model


def train_model(model, train_ds, val_ds):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        restore_best_weights=True
    )

    return model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])


def export_model_plot(model, path):
    to_file = os.path.join(path, 'model_plot.png')
    tf.keras.utils.plot_model(model, to_file=to_file, show_shapes=True, show_layer_names=True,
                              show_layer_activations=True, show_trainable=True)
    return to_file


def export_training_plot(history, path):
    to_file = os.path.join(path, 'training_plot.png')
    metrics, epoch = history.history, range(1, len(history.history['loss']) + 1)
    plt.rcParams.update({'font.size': 18})
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

    plt.savefig(to_file)
    return to_file


def main():
    os.makedirs(output_path, exist_ok=True)
    train_ds, val_ds, test_ds, label_names = import_data_set()
    model = build_model(label_names)
    history = train_model(model, train_ds, val_ds)

    model_plot_path = export_model_plot(model, path=output_path)
    training_plot_path = export_training_plot(history, path=output_path)
    print(f"Model plot saved to {model_plot_path} and training plot saved to {training_plot_path}")

    _, accuracy = model.evaluate(test_ds)
    print(f"Validation accuracy: {accuracy * 100}")


if __name__ == "__main__":
    main()
