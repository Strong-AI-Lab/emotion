"""Exports a TFRecord file from ARFF or raw audio data."""

import argparse
from pathlib import Path

import arff
import numpy as np
import soundfile
import tensorflow as tf

from emotion_recognition.dataset import parse_classification_annotations

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path,
                    help="ARFF file or file with list of filepaths.")
parser.add_argument('--labels', type=Path, help="Path to labels file.")
parser.add_argument('output', type=Path, help="Path to write TFRecord.")


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialise_example(name: str, features: np.ndarray, label: str):
    example = tf.train.Example(features=tf.train.Features(feature={
        'name': _bytes_feature(name),
        'features': _bytes_feature(features.tobytes()),
        'label': _bytes_feature(label),
        'features_shape': tf.train.Feature(int64_list=tf.train.Int64List(
            value=list(features.shape))),
        'features_dtype': _bytes_feature(np.dtype(features.dtype).str)
    }))
    return example.SerializeToString()


def main():
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(str(args.output))
    if args.input.suffix == '.arff':
        with open(args.input) as fid:
            data = arff.load(fid)
        for inst in data['data']:
            name = inst[0]
            label = inst[-1]
            features = np.array(inst[1:-1], dtype=np.float32)
            writer.write(serialise_example(name, features, label))
    else:
        with open(args.input) as fid:
            filenames = [x.strip() for x in fid.readlines()]
        if not args.labels:
            raise ValueError("Labels must be provided for raw audio dataset")
        label_dict = parse_classification_annotations(args.labels)
        for filename in filenames:
            Path(filename).stem
            audio, sr = soundfile.read(filename, dtype=np.float32)
            label = label_dict[Path(filename).stem]
            writer.write(serialise_example(name, audio, label))
    writer.close()


if __name__ == "__main__":
    main()
