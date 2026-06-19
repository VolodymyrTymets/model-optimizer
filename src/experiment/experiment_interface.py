from abc import ABC, abstractmethod

import tensorflow as tf


class IExperiment(ABC):
    @property
    def run(self, train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> None:
        pass
