from os import PathLike
from typing import Union

import numpy as np
import tensorflow as tf

from ..dataset import LabelledDataset


class TFRecordDataset(LabelledDataset):
    """A dataset contained in TFRecord files."""
    def __init__(self, file: Union[PathLike, str]):
        self.tf_dataset = tf.data.TFRecordDataset([str(file)])
        example = tf.train.Example()
        example.ParseFromString(next(iter(self.tf_dataset)).numpy())
        features = example.features.feature
        corpus = features['corpus'].bytes_list.value[0].decode()
        self.data_shape = tuple(features['features_shape'].int64_list.value)
        self.data_dtype = features[
            'features_dtype'].bytes_list.value[0].decode()
        super().__init__(corpus)

    def _create_data(self):
        self._x = []
        self._y = []
        for item in self.tf_dataset:
            example = tf.train.Example()
            example.ParseFromString(item.numpy())
            features = example.features.feature
            data = np.frombuffer(
                features['features'].bytes_list.value[0],
                dtype=self.dtype
            )
            data = np.reshape(data, self.data_shape)
            label = features['label'].bytes_list.value[0].decode()
            label_int = self.class_to_int(label)
            self._x.append(data)
            self._y.append(label_int)
        self._x = np.array(self._x, dtype=np.float32)
        self._y = np.array(self._y, dtype=np.float32)
