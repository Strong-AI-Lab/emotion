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

from emotion_recognition.dataset import (corpora, write_netcdf_dataset,
                                         parse_classification_annotations)


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


def log10(x: tf.Tensor):
    return tf.math.log(x) / tf.math.log(10.0)


def calculate_spectrogram(audio: tf.Tensor,
                          sample_rate: int = 16000,
                          channels: str = 'mean',
                          pre_emphasis: float = 0.95,
                          skip: float = 0,
                          length: float = 5,
                          window_size: float = 0.05,
                          window_shift: float = 0.025,
                          n_mels: int = 120,
                          clip: float = 60):
    """Calculates a spectrogram from a batch of time-domain signals.

    Args:
    -----
    audio: tf.Tensor
        The audio to process. Has shape (N, T, C), where N is the batch size,
        T is the length of the clip in samples, C is the number of channels
        (must be 1 if only one channel).
    sample_rate: int
        The sample rate, which must be constant across the batch.
    channels: str, one of {'mean', 'left', 'right', 'diff'}
        How to combine the channels. Default is 'mean'.
    pre_emphasis: float
        Amount of pre-emphasis to apply. Can be 0 for no pre-emphasis. Default
        is 0.95.
    skip: float
        Length in seconds to ignore from the start of the clip. Default is 0.
    length: float
        Desired length of the clip in seconds. Default is 5 seconds.
    window_size: float
        Length of window in seconds. Default is 0.05
    window_shift: float
        Window shift/stride in seconds. Default is 0.025
    n_mels: int
        Number of mel bands to calculate
    clip: float
        dB level below which to clip. Default is 60.

    Returns:
    --------
    spectrograms: tf.Tensor
        The batched spectrograms of shape (N, T', M), where N is the batch
        size, T' is the number of frames, M is n_mels.
    """
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

    # Channel fusion
    if channels == 'left':
        audio = audio[:, :, 0]
    elif channels == 'right':
        audio = audio[:, :, 1]
    elif channels == 'mean':
        audio = tf.reduce_mean(audio[:, :, :2], axis=-1)
    elif channels == 'diff':
        audio = audio[:, :, 0] - audio[:, :, 1]

    # Padding
    audio_len = audio.shape[1]
    if audio_len < length_samples:
        audio = tf.pad(audio, [[0, 0], [0, length_samples - audio_len]])

    # Pre-emphasis
    if pre_emphasis > 0:
        filt = audio[:, 1:] - pre_emphasis * audio[:, :-1]
        audio = tf.concat([audio[:, 0:1], filt], 1)

    # Spectrogram
    spectrogram = tf.abs(tf.signal.stft(audio, window_samples, stride_samples))
    # We need to ignore a frame because the auDeep increases the window shift
    # for some reason.
    spectrogram = spectrogram[..., :-1, :]

    # Mel spectrogram
    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram, sample_rate, n_mels, 0, 8000)

    # Calculate power spectrum in dB units and clip
    power = tf.square(mel_spectrogram)
    db = 10 * log10(power)
    max_db = tf.reduce_max(db, axis=[1, 2], keepdims=True)
    db_spectrogram = tf.maximum(db - max_db, -clip)

    # Scale spectrogram values to [-1, 1] as per auDeep.
    db_min = tf.reduce_min(db_spectrogram, axis=[1, 2], keepdims=True)
    db_max = tf.reduce_max(db_spectrogram, axis=[1, 2], keepdims=True)
    eps = db_max - db_min < 1e-4
    lower = db_spectrogram - db_min
    norm = 2 * (db_spectrogram - db_min) / (db_max - db_min) - 1
    db_spectrogram = tf.where(eps, lower, norm)
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
    parser.add_argument('--input', type=Path,
                        help="File containing list of WAV audio files.")

    parser.add_argument('--corpus', type=str, help="Corpus name.")
    parser.add_argument('--labels', type=Path,
                        help="Path to label annotations.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for processsing.")

    parser.add_argument('--netcdf', type=Path,
                        help="Output to NetCDF4 format.")
    parser.add_argument('--audeep', type=Path,
                        help="Output to NetCDF4 in audeep format.")

    parser.add_argument('--length', type=float, default=5,
                        help="Seconds of audio clip to take or pad.")
    parser.add_argument('--skip', type=float, default=0,
                        help="Seconds of initial audio to skip.")
    parser.add_argument('--clip', type=float, default=60,
                        help="Clip below this (negative) dB level.")
    parser.add_argument('--window_size', type=float, default=0.05,
                        help="Window size in seconds.")
    parser.add_argument('--window_shift', type=float, default=0.025,
                        help="Window shift in seconds.")
    parser.add_argument('--mel_bands', type=int, default=120,
                        help="Number of mel bands.")
    parser.add_argument('--pre_emphasis', type=float, default=0.95,
                        help="Pre-emphasis factor.")
    parser.add_argument(
        '--preview', type=int, help="Display nth spectrogram without writing "
        "to a file. A random spectrogram is displayed if p = -1. Overrides -t "
        "or -a."
    )
    parser.add_argument(
        '--channels', type=str, default='mean', help="Method for combining "
        "stereo channels. One of {mean, left, right, diff}. Default is mean."
    )
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
            audio, sample_rate=sample_rate, channels=args.channels,
            skip=args.skip, length=args.length, window_size=args.window_size,
            pre_emphasis=args.pre_emphasis, window_shift=args.window_shift,
            n_mels=args.mel_bands, clip=args.clip
        )
        spectrogram = spectrogram[0, :, :].numpy()

        print("Spectrogram for {}.".format(paths[idx]))

        plt.figure()
        plt.imshow(spectrogram)
        plt.show()
        return

    if not args.audeep and not args.netcdf:
        raise ValueError(
            "Must specify either --preview, --netcdf or --audeep options.")

    if args.labels:
        if not args.corpus:
            raise ValueError(
                "--corpus must be provided if labels are provided.")
        labels = parse_classification_annotations(args.labels)

    print("Processing spectrograms:")
    start_time = time.perf_counter()
    dataset, sample_rate = get_batched_audio(args.input, args.batch_size)
    specs = []
    for x in dataset:
        specs.append(calculate_spectrogram(
            x, sample_rate, channels=args.channels, skip=args.skip,
            length=args.length, window_size=args.window_size,
            pre_emphasis=args.pre_emphasis, window_shift=args.window_shift,
            n_mels=args.mel_bands, clip=args.clip
        ))
    with tf.device('/device:cpu:0'):
        specs = tf.concat(specs, 0)
    spectrograms = specs.numpy()
    print("Processed {} spectrograms in {:.4f}s".format(
        len(spectrograms), time.perf_counter() - start_time))

    filenames = [x.stem for x in paths]
    if args.audeep is not None:
        write_audeep_dataset(args.audeep, spectrograms, filenames,
                             args.mel_bands, labels, args.corpus)

        print("Wrote netCDF dataset to {}.".format(args.audeep))

    if args.netcdf is not None:
        slices = [spectrograms.shape[1] for _ in range(len(spectrograms))]
        spectrograms = np.concatenate(spectrograms)
        write_netcdf_dataset(
            args.netcdf, corpus=args.corpus, names=filenames, slices=slices,
            features=spectrograms, annotation_path=args.labels)

        print("Wrote netCDF dataset to {}.".format(args.netcdf))


if __name__ == "__main__":
    main()
