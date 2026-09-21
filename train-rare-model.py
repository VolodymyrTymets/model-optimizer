import tensorflow as tf
from keras.src.utils.file_utils import join
from tensorflow import keras
from tensorflow.keras import layers, models

from src.assets_service.assets_service import AssetsService
from src.data_set.data_set_cooker import DataSetCooker
from src.data_set.data_set_importer import DataSetImporter
from src.data_set.types import ArgumentationTypes
from src.definitions import DURATION, frame_length, hop_length, sr, labels, ASSETS_PATH
from src.experiment.experiment_types import ExperimentDataSetDetails, ExperimentDetails
from src.experiments.experiments import Experiments
from src.model_exporter import model_exporter
from src.model_exporter.model_exporter import ModelExporter
from src.model_schema.model_schema_types import LayerType, LossType, RegularizerType, OptimizerType, ActivationType
from src.model_validator.mode_validator import ModeValidator
from src.model_validator.model_record_label.model_record_label import ModelRecordLabeler
from src.model_validator.model_result_parser.model_result_parser import ModelResultParser
from src.utils.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.utils.audio_features.types import AFTypes
from src.experiment.experiment import Experiment
from src.utils.logger.logger_service import Logger

af_type = AFTypes.mfcc

experiment_details = ExperimentDetails(
    epochs=100,
    batch_size=32,
    layers=[LayerType.Conv],
    activation=[ActivationType.ReLU],
    units_range=[8, 1024],
    optimizer=[OptimizerType.Adam],
    regularizer=[RegularizerType.L1],
    loss=[LossType.SparseCategoricalCrossentropy]
)
data_set_details = ExperimentDataSetDetails(
    duration=DURATION, labels=labels, argumentation_types=[ArgumentationTypes.nothing],
    af_type=af_type
)
af_strategy = AFStrategyFactory(sr=sr, frame_length=frame_length,
                                hop_length=hop_length).create_strategy(af_type)


def init_experiment():
    experiments = Experiments()
    experiment = Experiment(
        details=experiment_details,
        data_set_details=data_set_details,
        af_strategy=af_strategy
    )
    return experiment


def preprocess_data():
    data_set_cooker = DataSetCooker(experiment_id=1, af_strategy=af_strategy)
    data_set_cooker.prepare(duration=DURATION, argumentation_types=[
        ArgumentationTypes.nothing,
    ])


def build_model(train_ds, val_ds, test_ds, label_names):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        tf.keras.layers.Normalization(name='normalization'),
        tf.keras.layers.Reshape((input_shape + (1,)), name="reshape_to_conv"),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(labels), activation='softmax')
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss')])

    return model


def train_model(model, train_ds, val_ds, test_ds, label_names):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        restore_best_weights=True
    )

    return model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping])


def main():
    experiment = init_experiment()
    experiment_id = experiment.get_experiment_id()
    preprocess_data()
    assets_service = AssetsService(experiment_id=experiment_id)
    model_exporter = ModelExporter(af_strategy=af_strategy)
    data_set_importer = DataSetImporter(experiment_id=experiment_id, duration=DURATION,
                                        af_strategy=af_strategy)
    train_ds, val_ds, test_ds, label_names = data_set_importer.import_data_set()
    model = build_model(train_ds, val_ds, test_ds, label_names)
    history = train_model(model, train_ds, val_ds, test_ds, label_names)

    mode_plot_path = model_exporter.export_model_plot(model, path=assets_service.get_model_path())
    training_plot_path = model_exporter.export_training_plot(history,
                                                             path=assets_service.get_model_path())
    print(f"Model plot saved to {mode_plot_path}"
          f" and plots saved to {training_plot_path}")

    mode_validator = ModeValidator(logger=Logger('ModeValidator'), af_strategy=af_strategy)

    record_acc, validation_acc, record_acc_dic = mode_validator.validate(model=model, data=test_ds,
                                                                         validation_records_path=assets_service.get_validation_records_path())

    print(f"Validation accuracy: {validation_acc}")
    print(f"Record accuracy: {record_acc}")
    print(f"Record accuracy per class: {record_acc_dic}")

    model_record_label_service = ModelRecordLabeler(ModelResultParser(af_strategy=af_strategy),
                                                    export_path=join(assets_service.get_data_set_path(),
                                                                     'labelable_records'))
    print(f"Labeling records...")
    print(f"From {assets_service.get_validation_records_path()}")
    for image_path, name in model_record_label_service.label_records(model=model,
                                                                     from_path=assets_service.get_validation_records_path()):
        print(f"Labeled {image_path}")


if __name__ == "__main__":
    main()
