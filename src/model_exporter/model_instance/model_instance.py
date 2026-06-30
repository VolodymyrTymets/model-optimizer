import tensorflow as tf

from src.utils.audio_features.types import AFTypes


class ModelInstance(tf.Module):
    def __init__(self, model, label_names, af_type: AFTypes, duration: str):
        self.model = model
        self.label_names = label_names
        self.input_shape =  model.input_shape[1:]
        self._af_type = af_type
        self._duration = duration

        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=self.input_shape, dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        x = tf.reshape(x, (-1,) + self.input_shape)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names,
                'label_names': self.label_names}

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.string, name="x")])
    def get_settings(self, x):
        return {
            "af_type": self._af_type.value,
            "duration": self._duration
        }
