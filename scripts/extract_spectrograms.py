#!/usr/bin/python3

"""Extracts spectrograms from several audio files and creates a netCFD4 file or
TFRecord file holding the data.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict

import netCDF4
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt

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


def write_audeep_dataset(path: Path,
                         spectrograms: np.ndarray,
                         filenames: List[str],
                         n_mels: int,
                         labels: Optional[Dict[str, str]] = None,
                         corpus: Optional[str] = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    tdim = spectrograms[0].shape[0]
    if any(x.shape[0] != tdim for x in spectrograms):
        tdim = 0
    dataset = netCDF4.Dataset(str(path), 'w')
    dataset.createDimension('instance', len(spectrograms))
    dataset.createDimension('fold', 0)
    dataset.createDimension('time', tdim)
    dataset.createDimension('freq', n_mels)

    # Although auDeep uses the actual path, we use just the name of the
    # audio clip.
    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = np.array(filenames)

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
    if labels:
        label_nominal[:] = np.array([labels[x] for x in filenames])
        emotions = list(corpora[corpus].emotion_map.values())
        label_numeric[:] = np.array([emotions.index(labels[x])
                                     for x in filenames])
    else:
        label_nominal[:] = np.zeros(spectrograms.shape[0], dtype=str)
        label_numeric[:] = np.zeros(spectrograms.shape[0], dtype=np.int64)

    features = dataset.createVariable('features', np.float32,
                                      ('instance', 'time', 'freq'))
    features[:, :, :] = spectrograms

    dataset.setncattr_string('feature_dims', '["time", "freq"]')
    dataset.setncattr_string('corpus', corpus or '')
    dataset.close()


def calculate_spectrogram(audio: tf.Tensor,
                          sample_rate: int = 16000,
                          channels: str = 'mean',
                          skip: float = 0,
                          length: float = 5,
                          window_size: float = 0.05,
                          window_shift: float = 0.025,
                          mel_bands: int = 120,
                          clip: float = 60):
    sample_rate_f = tf.cast(sample_rate, tf.float32)
    start_samples = tf.cast(tf.round(skip * sample_rate_f), tf.int32)
    length_samples = tf.cast(tf.round(length * sample_rate_f), tf.int32)
    window_samples = tf.cast(tf.round(window_size * sample_rate_f), tf.int32)
    stride_samples = tf.cast(tf.round(window_shift * sample_rate_f), tf.int32)

    if length_samples <= 0:
        length_samples = audio.shape[1] - start_samples

    # Clip and convert audio to float in range [-1, 1]
    audio = audio[..., start_samples:start_samples + length_samples, :]
    audio = tf.cast(audio, tf.float32) / 32768.0

    if channels == 'left':
        audio = audio[:, :, 0]
    elif channels == 'right':
        audio = audio[:, :, 1]
    elif channels == 'mean':
        audio = tf.reduce_mean(audio[:, :, :2], axis=-1)
    elif channels == 'diff':
        audio = audio[:, :, 0] - audio[:, :, 1]

    audio_len = audio.shape[1]
    if audio_len < length_samples:
        audio = tf.pad(audio, [[0, 0], [0, length_samples - audio_len]])

    spectrogram = tf.abs(tf.signal.stft(audio, window_samples, stride_samples))
    # We need to ignore a frame because the auDeep increases the window shift
    # for some reason.
    spectrogram = spectrogram[..., :-1, :]
    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram, sample_rate, mel_bands, 0, 8000)

    # Calculate power spectrum in dB units and clip
    power = tf.square(mel_spectrogram)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    max_spec = tf.reduce_max(log_spec, axis=[-2, -1], keepdims=True)
    db_spectrogram = tf.maximum(log_spec, max_spec - clip)

    # Scale spectrogram values to [-1, 1] as per auDeep.
    db_min = tf.reduce_min(db_spectrogram, axis=[-2, -2], keepdims=True)
    db_max = tf.reduce_max(db_spectrogram, axis=[-2, -1], keepdims=True)
    db_spectrogram = ((2.0 * db_spectrogram - db_max - db_min)
                      / (db_max - db_min))
    return db_spectrogram


def get_batched_audio(path: Path, batch_size: int = 128):
    def pad_tensor(t: tf.Tensor, shape):
        shape = tf.convert_to_tensor(list(shape))
        diff = shape - tf.shape(t)
        pad_dims = tf.transpose(tf.stack([tf.zeros_like(shape), diff]))
        return tf.pad(t, pad_dims)

    dataset = tf.data.TextLineDataset([str(path)])
    map_args = dict(deterministic=True,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(tf.io.read_file, **map_args)
    dataset = dataset.map(lambda x: tf.audio.decode_wav(x, desired_channels=2),
                          **map_args)
    sample_rate = next(iter(dataset))[1]
    dataset = dataset.map(lambda x: x[0], **map_args)
    max_len = max(x.shape[0] for x in dataset)
    dataset = dataset.map(lambda x: pad_tensor(x, (max_len, 2)), **map_args)
    dataset = dataset.batch(batch_size)
    return dataset, sample_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path,
                        help="File containing list of WAV audio files.")

    parser.add_argument('-n', '--corpus', type=str, help="Corpus name.")
    parser.add_argument('--labels', type=Path,
                        help="Path to label annotations.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for processsing.")

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
    parser.add_argument('-p', '--preview', type=int, default=None,
                        help="Display nth spectrogram without writing to a "
                        "file. A random spectrogram is displayed if p = -1. "
                        "Overrides -t or -a.")
    parser.add_argument('--channels', type=str, default='mean',
                        help="Method for combining stereo channels. One of "
                        "{mean, left, right, diff}. Default is mean.")
    args = parser.parse_args()

    with open(args.input) as fid:
        paths = sorted(Path(x.strip()) for x in fid)

    if args.preview is not None:
        idx = args.preview
        if args.preview == -1:
            idx = np.random.randint(len(paths))

        wav = tf.io.read_file(str(paths[idx]))
        audio, sample_rate = tf.audio.decode_wav(wav)
        audio = tf.expand_dims(audio, 0)
        spectrogram = calculate_spectrogram(
            audio, sample_rate, args.channels, args.skip, args.length,
            args.window_size, args.window_shift, args.mel_bands, args.clip
        )
        spectrogram = spectrogram[0, :, :].numpy()

        print("Spectrogram for {}.".format(paths[idx]))

        plt.figure()
        plt.imshow(spectrogram)
        plt.show()
        return

    if not args.audeep and not args.tfrecord:
        raise ValueError(
            "Must specify either --preview, --tfrecord or --audeep options.")

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
    dataset, sample_rate = get_batched_audio(args.input, args.batch_size)
    specs = []
    for x in dataset:
        specs.append(calculate_spectrogram(
            x, sample_rate, channels=args.channels, skip=args.skip,
            length=args.length, window_size=args.window_size,
            window_shift=args.window_shift, mel_bands=args.mel_bands,
            clip=args.clip
        ))
    specs = tf.concat(specs, 0)
    spectrograms = specs.numpy()
    print("Processed {} spectrograms in {:.4f}s".format(
        len(spectrograms), time.perf_counter() - start_time))

    if args.audeep is not None:
        filenames = [x.stem for x in paths]
        write_audeep_dataset(args.audeep, spectrograms, filenames,
                             args.mel_bands, labels, args.corpus)

        print("Wrote netCDF dataset to {}.".format(args.audeep))

    if args.tfrecord is not None:
        with tf.io.TFRecordWriter(str(args.tfrecord)) as writer:
            for i in range(len(paths)):
                name = paths[i].stem
                label = labels[name]
                writer.write(serialise_example(
                    name, spectrograms[i], label,
                    speaker_idx[i] if args.corpus else 0, args.corpus or ''
                ))
        print("Wrote TFRecord to {}.".format(args.tfrecord))


if __name__ == "__main__":
    main()
