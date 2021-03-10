import time
import warnings
from pathlib import Path
from typing import List, Optional

import click
from click.types import Choice
import joblib
import librosa
import netCDF4
import numpy as np
from click_option_group import optgroup, RequiredAnyOptionGroup
from emorec.dataset import get_audio_paths, write_netcdf_dataset
from emorec.utils import PathlibPath
from matplotlib import pyplot as plt


def write_audeep_dataset(path: Path,
                         spectrograms: np.ndarray,
                         filenames: List[str],
                         n_mels: int,
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

    label_nominal = dataset.createVariable('label_nominal', str, ('instance',))
    label_numeric = dataset.createVariable('label_numeric', np.int64,
                                           ('instance',))
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
    warnings.simplefilter('ignore', UserWarning)
    n_fft = 2**int(np.ceil(np.log2(window_samples)))
    melspec = librosa.feature.melspectrogram(
        audio, n_mels=n_mels, sr=sr, n_fft=n_fft, hop_length=stride_samples,
        win_length=window_samples, fmin=fmin, fmax=fmax
    )
    warnings.simplefilter('default', UserWarning)
    db_spectrogram = librosa.power_to_db(melspec, ref=np.max, top_db=clip)

    return db_spectrogram.T


@click.command()
@click.argument('corpus', type=str)
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@optgroup.group('Output format', cls=RequiredAnyOptionGroup)
@optgroup.option('--netcdf', type=Path, help="Write dataset")
@optgroup.option('--audeep', type=Path, help="Write auDeep dataset")
@optgroup.option('--preview', type=int, help="Preview spectrogram")
@optgroup.group('Spectrogram options')
@optgroup.option('--length', type=float, help="Optional max clip length")
@optgroup.option('--skip', type=float, default=0,
                 help="Optional amount to skip, in seconds")
@optgroup.option('--clip', type=float, help="Optional minimum power in dB")
@optgroup.option('--window_size', type=float, default=0.025, show_default=True,
                 help="Window size in seconds")
@optgroup.option('--window_shift', type=float, default=0.010,
                 show_default=True, help="Window shift in seconds")
@optgroup.option('--mel_bands', type=int, default=240, show_default=True,
                 help="Number of mel bands")
@optgroup.option('--pre_emphasis', type=float, default=0.95, show_default=True,
                 help="Pre-emphasis applied before processing")
@optgroup.option('--channels', type=Choice(['left', 'right', 'mean', 'diff']),
                 default='mean', show_default=True)
@optgroup.option('--fmin', type=float, default=0, show_default=True,
                 help="Min mel frequency")
@optgroup.option('--fmax', type=float, default=8000, show_default=True,
                 help="Max mel frequency")
@click.option('--debug', is_flag=True, help="Debug mode (no multiprocessing)")
def main(corpus: str, input: Path, netcdf: Path, audeep: Path, preview: int,
         length: float, skip: float, clip: float, window_size: float,
         window_shift: float, mel_bands: int, pre_emphasis: float,
         channels: str, fmin: float, fmax: float, debug: bool):
    """Extracts spectrograms from audio files listed in INPUT file and
    creates a netCFD4 dataset or auDeep dataset holding the data. CORPUS
    specifies the corpus.
    """

    paths = get_audio_paths(input)

    if preview is not None:
        idx = preview if preview > -1 \
            else np.random.default_rng().integers(len(paths))

        spectrogram = calculate_spectrogram(
            paths[idx], channels=channels, skip=skip, length=length,
            window_size=window_size, pre_emphasis=pre_emphasis,
            window_shift=window_shift, n_mels=mel_bands, clip=clip, fmin=fmin,
            fmax=fmax
        )
        print(f"Spectrogram for {paths[idx]}.")

        plt.figure()
        plt.imshow(spectrogram)
        plt.show()
        return

    if not audeep and not netcdf:
        raise ValueError(
            "Must specify either --preview, --netcdf or --audeep options.")

    print("Processing spectrograms:")
    start_time = time.perf_counter()
    specs = joblib.Parallel(n_jobs=1 if debug else -1, verbose=1)(
        joblib.delayed(calculate_spectrogram)(
            path, channels=channels, skip=skip, length=length,
            window_size=window_size, pre_emphasis=pre_emphasis,
            window_shift=window_shift, n_mels=mel_bands, clip=clip, fmin=fmin,
            fmax=fmax
        ) for path in paths
    )
    total_time = time.perf_counter() - start_time
    print(f"Processed {len(specs)} spectrograms in {total_time:.4f}s")

    filenames = [x.stem for x in paths]
    if audeep is not None:
        stacked = np.stack(specs)
        amin = np.min(stacked, axis=(1, 2), keepdims=True)
        amax = np.max(stacked, axis=(1, 2), keepdims=True)
        stacked = 2 * (stacked - amin) / (amax - amin) - 1
        write_audeep_dataset(audeep, stacked, filenames, mel_bands,
                             corpus)
        print(f"Wrote auDeep dataset to {audeep}")

    if netcdf is not None:
        slices = [len(x) for x in specs]
        specs = np.concatenate(specs)
        feature_names = [f'meldB{i + 1}' for i in range(mel_bands)]
        write_netcdf_dataset(
            netcdf, corpus=corpus, names=filenames, slices=slices,
            features=specs, feature_names=feature_names
        )
        print(f"Wrote netCDF dataset to {netcdf}")


if __name__ == "__main__":
    main()
