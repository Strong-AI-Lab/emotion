from pathlib import Path

import click
import numpy as np
import soundfile
from emorec.dataset import get_audio_paths, write_netcdf_dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument("corpus", type=str)
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
def main(corpus: str, input: Path, output: Path):
    """Creates a NetCDF dataset containing the raw audio from files
    listed in INPUT file and writes to OUTPUT file. The CORPUS argument
    specified the corpus.
    """

    filenames = get_audio_paths(input)

    print(f"Processing {len(filenames)} audio files.")
    audio_list = []
    for filename in filenames:
        audio, sr = soundfile.read(filename, always_2d=True, dtype=np.float32)
        if sr != 16000:
            print(f"Sample rate of {filename} != 16000, skipping.")
            continue
        audio = np.mean(audio, axis=1)
        audio = np.expand_dims(audio, axis=1)
        audio_list.append(audio)
    slices = [len(x) for x in audio_list]
    audio_arr = np.concatenate(audio_list)
    print("Time:")
    print(f"\ttotal: {sum(slices) / 16000:.3f} s")
    print(f"\tmin: {min(slices) / 16000:.3f} s")
    print(f"\tmax: {max(slices) / 16000:.3f} s")
    print(f"\tmean: {np.mean(slices) / 16000:.3f} s")
    print(f"\tstd: {np.std(slices) / 16000:.3f} s")

    names = [f.stem for f in filenames]
    write_netcdf_dataset(
        output, corpus=corpus, names=names, slices=slices, features=audio_arr
    )
    print(f"Wrote NetCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
