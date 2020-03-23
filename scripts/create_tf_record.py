import argparse
from pathlib import Path

import arff
import numpy as np
import soundfile
import tensorflow as tf

from python.dataset import parse_classification_annotations

parser = argparse.ArgumentParser()
parser.add_argument('input', help="ARFF file or file with list of filepaths.")
parser.add_argument('--labels', type=str, help="Path to labels file.")
parser.add_argument('output', help="Path to write TFRecord.")


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialise_arff_example(name: str, features: np.ndarray, label: int):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'name': _bytes_feature(name),
        'features': _bytes_feature(features.tobytes())
    }))
    return example.SerializeToString()


def serialise_audio_example(audio: np.ndarray, label: int):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'raw_audio': _bytes_feature(audio.tobytes())
    }))
    return example.SerializeToString()


def main():
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(args.output)
    if args.input.endswith('.arff'):
        with open(args.input) as fid:
            data = arff.load(fid)
        for inst in data['data']:
            name = inst[0]
            label = inst[-1]
            features = np.array(inst[1:-1], dtype=np.float32)
            writer.write(serialise_arff_example(name, features, label))
    else:
        with open(args.input) as fid:
            filenames = [x.strip() for x in fid.readlines()]
        if not args.labels:
            raise ValueError("--labels must be provided for raw audio dataset")
        label_dict = parse_classification_annotations(args.labels)
        labels = sorted(set(label_dict.values()))
        for filename in filenames:
            audio, sr = soundfile.read(filename, dtype=np.float32)
            audio = (audio + 1) / 2
            audio = np.pad(audio, (0, 640 - audio.shape[0] % 640))
            label = label_dict[Path(filename).stem]
            label = labels.index(label)
            writer.write(serialise_audio_example(audio, label))
    writer.close()


if __name__ == "__main__":
    main()
