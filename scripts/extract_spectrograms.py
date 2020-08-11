#!/usr/bin/python3

"""Extracts spectrograms from several audio files and creates a netCFD4 file or
TFRecord file holding the data.
"""

import argparse
import time
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
        'features_dtype': _bytes_feature(
            np.dtype(features.dtype).str.encode()),
        'corpus': _bytes_feature(corpus)
    }))
    return example.SerializeToString()


def get_spectrogram(file: str,
                    channels: str = 'mean',
                    skip: float = 0,
                    length: float = 5,
                    window_size: float = 0.05,
                    window_shift: float = 0.025,
                    mel_bands: int = 120,
                    clip: float = 60):
    wav = tf.io.read_file(file)
    audio, sample_rate = tf.audio.decode_wav(wav)

    sample_rate_f = tf.cast(sample_rate, tf.float32)
    start = tf.cast(tf.round(skip * sample_rate_f), tf.int32)
    length = tf.cast(tf.round(length * sample_rate_f), tf.int32)
    window_samples = tf.cast(tf.round(window_size * sample_rate_f), tf.int32)
    stride_samples = tf.cast(tf.round(window_shift * sample_rate_f), tf.int32)

    length = tf.where(length <= 0, tf.shape(audio)[0] - start, length)

    # Clip and convert audio to float in range [-1, 1]
    audio = audio[start:start + length]
    audio = tf.cast(audio, tf.float32) / 32768.0
    audio_len = tf.shape(audio)[0]

    if channels == 'left':
        audio = audio[:, 0]
    elif channels == 'right':
        audio = audio[:, 1]
    elif channels == 'mean':
        audio = tf.reduce_mean(audio[:, :2], axis=1)
    elif channels == 'diff':
        audio = audio[:, 0] - audio[:, 1]

    if audio_len < length:
        audio = tf.pad(audio, [[0, length - audio_len]])

    spectrogram = tf.abs(tf.signal.stft(audio, window_samples, stride_samples))
    # We need to ignore a frame because the auDeep increases the window shift
    # for some reason.
    spectrogram = spectrogram[..., :-1, :]
    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram, sample_rate, mel_bands, 0, 8000)

    # Calculate power spectrum in dB units and clip
    power = tf.square(mel_spectrogram)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    db_spectrogram = tf.maximum(log_spec,
                                tf.reduce_max(log_spec, axis=[-2, -1]) - clip)
    # Scale spectrogram values to [-1, 1] as per auDeep.
    db_min = tf.reduce_min(db_spectrogram)
    db_max = tf.reduce_max(db_spectrogram)
    db_spectrogram = ((2.0 * db_spectrogram - db_max - db_min)
                      / (db_max - db_min))
    return db_spectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=Path,
                        help="File containing list of WAV audio files.")

    parser.add_argument('-n', '--corpus', type=str, help="Corpus name.")

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

    with open(args.input_files) as fid:
        paths = sorted(Path(x.strip()) for x in fid)

    if args.pretend is not None:
        pretend_idx = args.pretend
        if args.pretend == -1:
            pretend_idx = np.random.randint(len(paths))
        plt.figure()
        spectrogram = get_spectrogram(
            str(paths[pretend_idx]), args.channels, args.skip, args.length,
            args.window_size, args.window_shift, args.mel_bands, args.clip
        )
        spectrogram = spectrogram.numpy()
        plt.imshow(spectrogram)
        plt.show()
        return

    if args.audeep is None and args.tfrecord is None:
        raise ValueError(
            "Must specify either --pretend, --tfrecord or --audeep options.")

    if args.labels:
        if not args.corpus:
            raise ValueError(
                "--corpus must be provided if labels are provided.")
        labels = parse_classification_annotations(args.labels)

    if args.corpus:
        # For labelling the speaker cross-validation folds
        speakers = corpora[args.corpus].speakers
        get_speaker = corpora[args.corpus].get_speaker
        speaker_idx = [speakers.index(get_speaker(x.stem)) for x in paths]

    print("Processing spectrograms:")
    start_time = time.perf_counter()
    spectrograms = []
    for path in tqdm(paths, desc="Analysing spectrograms", unit='sp'):
        spectrogram = get_spectrogram(
            str(path), args.channels, args.skip, args.length,
            args.window_size, args.window_shift, args.mel_bands, args.clip
        )
        spectrograms.append(spectrogram.numpy())
    spectrograms = np.array(spectrograms)
    print("Processed {} spectrograms in {:.4f}s".format(
        len(spectrograms), time.perf_counter() - start_time))

    if args.audeep is not None:
        args.audeep.parent.mkdir(parents=True, exist_ok=True)

        tdim = spectrograms[0].shape[0] if args.length else 0
        dataset = netCDF4.Dataset(str(args.audeep), 'w')
        dataset.createDimension('instance', len(spectrograms))
        dataset.createDimension('fold', 0)
        dataset.createDimension('time', tdim)
        dataset.createDimension('freq', args.mel_bands)

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

        dataset.createVariable('cv_folds', np.int64, ('instance', 'fold'))

        label_nominal = dataset.createVariable('label_nominal', str,
                                               ('instance',))
        label_numeric = dataset.createVariable('label_numeric', np.int64,
                                               ('instance',))
        if args.labels:
            label_nominal[:] = np.array([labels[x.stem] for x in paths])
            emotions = list(corpora[args.corpus].emotion_map.values())
            label_numeric[:] = np.array([emotions.index(labels[x.stem])
                                         for x in paths])
        else:
            label_nominal[:] = np.zeros(spectrograms.shape[0], dtype=str)
            label_numeric[:] = np.zeros(spectrograms.shape[0], dtype=np.int64)

        features = dataset.createVariable('features', np.float32,
                                          ('instance', 'time', 'freq'))
        features[:, :, :] = spectrograms

        dataset.setncattr_string('feature_dims', '["time", "freq"]')
        dataset.setncattr_string('corpus', args.corpus or '')
        dataset.close()

        print("Wrote netCDF dataset to {}.".format(str(args.audeep)))

    if args.tfrecord is not None:
        with tf.io.TFRecordWriter(str(args.tfrecord)) as writer:
            for i in range(len(paths)):
                name = paths[i].stem
                label = labels[name]
                writer.write(serialise_example(
                    name, spectrograms[i], label,
                    speaker_idx[i] if args.corpus else 0, args.corpus or ''
                ))


if __name__ == "__main__":
    main()
