#!/usr/bin/python3

"""Extracts spectrograms from several audio files and creates a netCFD4 file or
TFRecord file holding the data.
"""

import argparse
from os import PathLike
from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from tqdm import tqdm

from emotion_recognition.dataset import (corpora,
                                         parse_classification_annotations)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialise_example(name: str, features: np.ndarray, label: str, fold: int,
                      corpus: str):
    example = tf.train.Example(features=tf.train.Features(feature={
        'name': _bytes_feature(name),
        'features': _bytes_feature(features.tobytes()),
        'label': _bytes_feature(label),
        'fold': _int64_feature(fold),
        'features_shape': tf.train.Feature(int64_list=tf.train.Int64List(
            value=list(features.shape))),
        'features_dtype': _bytes_feature(np.dtype(features.dtype).str.encode()),
        'corpus': _bytes_feature(corpus)
    }))
    return example.SerializeToString()


def get_spectrogram(file: PathLike,
                    channels: str,
                    skip: float,
                    length: float,
                    window_size: float,
                    window_shift: float,
                    mel_bands: int,
                    clip: float):
    audio = tfio.audio.AudioIOTensor(str(file))
    if audio.shape[1] > 2:
        raise ValueError("Only stereo or mono audio is supported, "
                         "{} channels found.".format(audio.shape[1]))
    sample_rate = int(audio.rate.numpy())
    start = int(skip * sample_rate)
    length = int(length * sample_rate)

    # Convert audio to float in range [-1, 1]
    audio = tf.cast(audio[start:start + length], tf.float32) / 32768.0
    if channels == 'left':
        audio = audio[:, 0]
    elif channels == 'right':
        audio = audio[:, 1]
    elif channels == 'mean':
        audio = tf.reduce_mean(audio, axis=1)
    elif channels == 'diff':
        audio = audio[:, 0] - audio[:, 1]

    if audio.shape[0] < length:
        audio = tf.pad(audio, [[0, length - audio.shape[0]]])

    spectrogram = tfio.experimental.audio.spectrogram(
        audio, nfft=2048, window=int(window_size * sample_rate),
        stride=int(window_shift * sample_rate)
    )
    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram, sample_rate, mel_bands, 0, 8000)

    # Calculate power spectrum in dB units and clip
    db_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram,
                                                     clip)
    # Scale spectrogram values to [-1, 1] as per auDeep.
    db_min = tf.reduce_min(db_spectrogram)
    db_max = tf.reduce_max(db_spectrogram)
    db_spectrogram = ((2.0 * db_spectrogram - db_max - db_min)
                      / (db_max - db_min))
    return db_spectrogram.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path,
                        help="Directory containing WAV audio.")
    parser.add_argument('-n', '--corpus', type=str, required=True,
                        help="Corpus name.")

    parser.add_argument('-t', '--tfrecord', type=Path,
                        help="Output to TFRecord format.")
    parser.add_argument('-a', '--audeep', type=Path,
                        help="Output to NetCDF4 in audeep format.")

    parser.add_argument('-l', '--length', type=float, default=5,
                        help="Seconds of audio clip to take or pad.")
    parser.add_argument('-s', '--skip', type=float, default=0,
                        help="Seconds of initial audio to skip.")
    parser.add_argument('-c', '--clip', type=float, default=60,
                        help="Clip below this (negative) dB level.")
    parser.add_argument('-w', '--window_size', type=float, default=0.05,
                        help="Window size in seconds.")
    parser.add_argument('-x', '--window_shift', type=float, default=0.025,
                        help="Window shift in seconds.")
    parser.add_argument('-m', '--mel_bands', type=int, default=120,
                        help="Number of mel bands.")
    parser.add_argument('-p', '--pretend', type=int, default=None,
                        help="Display nth spectrogram without writing to a "
                        "file. A random spectrogram is displayed if p = -1. "
                        "Overrides -t or -a.")
    parser.add_argument('--channels', type=str, default='mean',
                        help="Method for combining stereo channels. One of "
                        "{mean, left, right, diff}. Default is mean.")
    parser.add_argument('--labels', type=Path,
                        help="Path to label annotations.")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob('*.wav'))

    if args.pretend is not None:
        pretend_idx = args.pretend
        if args.pretend == -1:
            pretend_idx = np.random.randint(len(paths))
        plt.figure()
        spectrogram = get_spectrogram(
            paths[pretend_idx], args.channels, args.skip, args.length,
            args.window_size, args.window_shift, args.mel_bands, args.clip
        )
        plt.imshow(spectrogram)
        plt.show()
        return

    if args.audeep is None and args.tfrecord is None:
        raise ValueError("Must specify either -p, -t or -a options.")

    if args.labels:
        labels = parse_classification_annotations(args.labels)

    # For labelling the speaker cross-validation folds
    speakers = corpora[args.corpus].speakers
    get_speaker = corpora[args.corpus].get_speaker
    speaker_idx = [speakers.index(get_speaker(x.stem)) for x in paths]

    print("Processing spectrograms:")
    spectrograms = []
    for file in tqdm(paths, desc="Analysing spectrograms", unit='sp'):
        spectrogram = get_spectrogram(
            file, args.channels, args.skip, args.length,
            args.window_size, args.window_shift, args.mel_bands, args.clip
        )
        spectrograms.append(spectrogram)
    spectrograms = np.array(spectrograms)

    if args.audeep is not None:
        args.audeep.parent.mkdir(parents=True, exist_ok=True)

        dataset = netCDF4.Dataset(str(args.audeep), 'w')
        dataset.createDimension('instance', spectrograms.shape[0])
        dataset.createDimension('fold', len(speakers))
        dataset.createDimension('time', spectrograms.shape[1])
        dataset.createDimension('freq', spectrograms.shape[2])

        # Although auDeep uses the actual path, we use just the name of the
        # audio clip.
        filename = dataset.createVariable('filename', str, ('instance',))
        filename[:] = np.array([x.stem for x in paths])

        # Partition and chunk are unused, set them to defaults.
        chunk_nr = dataset.createVariable('chunk_nr', np.int64, ('instance',))
        chunk_nr[:] = np.zeros(spectrograms.shape[0], dtype=np.int64)
        partition = dataset.createVariable('partition', np.float64,
                                           ('instance',))
        partition[:] = np.zeros(spectrograms.shape[0], dtype=np.float64)

        cv_folds = dataset.createVariable('cv_folds', np.int64, ('instance',
                                                                 'fold'))
        cv_folds[:, :] = np.zeros((spectrograms.shape[0], len(speakers)))
        for i in range(len(paths)):
            cv_folds[i, speaker_idx[i]] = 1

        label_nominal = dataset.createVariable('label_nominal', str,
                                               ('instance',))
        label_numeric = dataset.createVariable('label_numeric', np.int64,
                                               ('instance',))
        if args.labels:
            emotions = list(corpora[args.corpus].emotion_map.values())
            label_nominal[:] = np.array([labels[x.stem] for x in paths])
            label_numeric[:] = np.array([emotions.index(labels[x.stem])
                                         for x in paths])
        else:
            label_numeric[:] = np.zeros(spectrograms.shape[0], dtype=np.int64)

        features = dataset.createVariable('features', np.float32,
                                          ('instance', 'time', 'freq'))
        features[:, :, :] = spectrograms

        dataset.setncattr_string('feature_dims', '["time", "freq"]')
        dataset.setncattr_string('corpus', args.corpus)
        dataset.close()

        print("Wrote netCDF dataset to {}.".format(str(args.audeep)))

    if args.tfrecord is not None:
        with tf.io.TFRecordWriter(str(args.tfrecord)) as writer:
            for i in range(len(paths)):
                name = paths[i].stem
                label = labels[name]
                writer.write(serialise_example(name, spectrograms[i], label,
                                               speaker_idx[i], args.corpus))


if __name__ == "__main__":
    main()
