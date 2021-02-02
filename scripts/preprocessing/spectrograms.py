"""Extracts spectrograms from several audio files and creates a netCFD4 file or
TFRecord file holding the data.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import joblib
import librosa
import netCDF4
import numpy as np
from emotion_recognition.dataset import (corpora, get_audio_paths,
                                         parse_classification_annotations,
                                         write_netcdf_dataset)
from matplotlib import pyplot as plt


def write_audeep_dataset(path: Path,
                         spectrograms: np.ndarray,
                         filenames: List[str],
                         n_mels: int,
                         labelpath: Optional[Path] = None,
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
    if labelpath:
        labels = parse_classification_annotations(labelpath)
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


def calculate_spectrogram(path: Path,
                          channels: str = 'mean',
                          pre_emphasis: float = 0.95,
                          skip: float = 0,
                          length: Optional[float] = None,
                          window_size: float = 0.025,
                          window_shift: float = 0.01,
                          n_mels: int = 240,
                          clip: Optional[float] = None,
                          fmin: float = 0,
                          fmax: float = 8000):
    audio, sr = librosa.core.load(path, sr=None, mono=False, offset=skip,
                                  duration=length)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, 0)

    window_samples = int(window_size * sr)
    stride_samples = int(window_shift * sr)

    # Channel fusion
    if channels == 'left':
        audio = audio[0]
    elif channels == 'right':
        audio = audio[1]
    elif channels == 'mean':
        audio = np.mean(audio[:2], axis=0)
    elif channels == 'diff':
        audio = audio[0] - audio[1]

    # Padding
    if length is not None and length > 0:
        length_samples = int(length * sr)
        if len(audio) < length_samples:
            audio = np.pad(audio, (0, length_samples - len(audio)))
        assert len(audio) == length_samples

    # Pre-emphasis
    if pre_emphasis > 0:
        audio = librosa.effects.preemphasis(audio, pre_emphasis)

    # Mel spectrogram
    n_fft = 2**int(np.math.ceil(np.log2(window_samples)))
    melspec = librosa.feature.melspectrogram(
        audio, n_mels=n_mels, sr=sr, n_fft=n_fft, hop_length=stride_samples,
        win_length=window_samples, fmin=fmin, fmax=fmax
    )
    db_spectrogram = librosa.power_to_db(melspec, ref=np.max, top_db=clip)
    db_min = db_spectrogram.min()
    db_max = db_spectrogram.max()
    db_spectrogram = 2 * (db_spectrogram - db_min) / (db_max - db_min) - 1
    return db_spectrogram.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path,
                        help="File containing list of WAV audio files.")

    parser.add_argument('--corpus', type=str, help="Corpus name.")
    parser.add_argument('--labels', type=Path,
                        help="Path to label annotations.")

    parser.add_argument('--netcdf', type=Path,
                        help="Output to NetCDF4 format.")
    parser.add_argument('--audeep', type=Path,
                        help="Output to NetCDF4 in audeep format.")
    parser.add_argument(
        '--preview', type=int, help="Display nth spectrogram without writing "
        "to a file. A random spectrogram is displayed if p = -1."
    )

    parser.add_argument('--length', type=float,
                        help="Seconds of audio clip to take or pad.")
    parser.add_argument('--skip', type=float, default=0,
                        help="Seconds of initial audio to skip.")
    parser.add_argument('--clip', type=float,
                        help="Clip below this (negative) dB level.")
    parser.add_argument('--window_size', type=float, default=0.025,
                        help="Window size in seconds.")
    parser.add_argument('--window_shift', type=float, default=0.01,
                        help="Window shift in seconds.")
    parser.add_argument('--mel_bands', type=int, default=240,
                        help="Number of mel bands.")
    parser.add_argument('--pre_emphasis', type=float, default=0.95,
                        help="Pre-emphasis factor.")
    parser.add_argument(
        '--channels', type=str, default='mean', help="Method for combining "
        "stereo channels. One of {mean, left, right, diff}. Default is mean."
    )
    parser.add_argument('--fmin', type=float, default=0,
                        help="Minimum mel-frequency.")
    parser.add_argument('--fmax', type=float, default=8000,
                        help="Minimum mel-frequency.")
    args = parser.parse_args()

    paths = get_audio_paths(args.input)

    if args.preview is not None:
        idx = args.preview if args.preview > -1 \
            else np.random.randint(len(paths))

        spectrogram = calculate_spectrogram(
            paths[idx], channels=args.channels,
            skip=args.skip, length=args.length, window_size=args.window_size,
            pre_emphasis=args.pre_emphasis, window_shift=args.window_shift,
            n_mels=args.mel_bands, clip=args.clip, fmin=args.fmin,
            fmax=args.fmax
        )
        print("Spectrogram for {}.".format(paths[idx]))

        plt.figure()
        plt.imshow(spectrogram)
        plt.show()
        return

    if not args.audeep and not args.netcdf:
        raise ValueError(
            "Must specify either --preview, --netcdf or --audeep options.")

    print("Processing spectrograms:")
    start_time = time.perf_counter()
    specs = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(calculate_spectrogram)(
            path, channels=args.channels, skip=args.skip, length=args.length,
            window_size=args.window_size, pre_emphasis=args.pre_emphasis,
            window_shift=args.window_shift, n_mels=args.mel_bands,
            clip=args.clip, fmin=args.fmin, fmax=args.fmax
        ) for path in paths
    )
    print("Processed {} spectrograms in {:.4f}s".format(
        len(specs), time.perf_counter() - start_time))

    if args.labels and not args.corpus:
        raise ValueError("--corpus must be provided if labels are provided.")

    filenames = [x.stem for x in paths]
    if args.audeep is not None:
        write_audeep_dataset(args.audeep, np.stack(specs), filenames,
                             args.mel_bands, args.labels, args.corpus)

        print("Wrote auDeep-specific dataset to {}.".format(args.audeep))

    if args.netcdf is not None:
        slices = [len(x) for x in specs]
        spectrograms = np.concatenate(specs)
        write_netcdf_dataset(
            args.netcdf, corpus=args.corpus, names=filenames, slices=slices,
            features=spectrograms, annotation_path=args.labels)

        print("Wrote netCDF dataset to {}".format(args.netcdf))


if __name__ == "__main__":
    main()
