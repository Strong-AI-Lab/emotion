"""Implementation of the model architecture from [1]_.

References
----------
.. [1] Z. Aldeneh and E. Mower Provost, 'Using regional saliency for
       speech emotion recognition', in 2017 IEEE International
       Conference on Acoustics, Speech and Signal Processing (ICASSP),
       Mar. 2017, pp. 2741-2745, doi: 10.1109/ICASSP.2017.7952655.
"""

import keras

from ertk.tensorflow.classification import tf_classification_metrics

__all__ = ["model", "Aldeneh2017Model"]


class Aldeneh2017Model(keras.Model):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.norm = keras.layers.LayerNormalization(name="norm")
        self.norm.trainable = False
        self.conv8 = keras.layers.Conv1D(
            384, 8, activation="relu", kernel_initializer="he_normal", name="conv8"
        )
        self.conv16 = keras.layers.Conv1D(
            384, 16, activation="relu", kernel_initializer="he_normal", name="conv16"
        )
        self.conv32 = keras.layers.Conv1D(
            384, 32, activation="relu", kernel_initializer="he_normal", name="conv32"
        )
        self.conv64 = keras.layers.Conv1D(
            384, 64, activation="relu", kernel_initializer="he_normal", name="conv64"
        )
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(
                    1024,
                    activation="relu",
                    kernel_initializer="he_normal",
                    name="dense_1",
                ),
                keras.layers.Dense(
                    1024,
                    activation="relu",
                    kernel_initializer="he_normal",
                    name="dense_2",
                ),
                keras.layers.Dense(
                    self.n_classes,
                    activation="softmax",
                    kernel_initializer="he_normal",
                    name="emotion_prediction",
                ),
            ]
        )
        self.maxpool = keras.layers.GlobalMaxPool1D(name="maxpool")

    def call(self, inputs):
        inputs = self.norm(inputs)
        x = self.conv8(inputs)
        c1 = self.maxpool(x)

        x = self.conv16(inputs)
        c2 = self.maxpool(x)

        x = self.conv32(inputs)
        c3 = self.maxpool(x)

        x = self.conv64(inputs)
        c4 = self.maxpool(x)

        x = keras.layers.concatenate([c1, c2, c3, c4], name="concatenate")
        x = self.mlp(x)
        return x


def model(
    n_features: int, n_classes: int, learning_rate: float = 0.001
) -> Aldeneh2017Model:
    model = Aldeneh2017Model(n_features, n_classes)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=tf_classification_metrics(),
        weighted_metrics=[],
    )
    return model
