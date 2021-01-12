"""Creates a NetCDF dataset containing the raw audio and labels."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile
from emotion_recognition.dataset import get_audio_paths, write_netcdf_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    filenames = get_audio_paths(args.input)

    print("Processing {} audio files.".format(len(filenames)))
    audio_arr = []
    for filename in filenames:
        audio, sr = soundfile.read(filename, always_2d=True, dtype=np.float32)
        if sr != 16000:
            sys.stderr.write(
                "Sample rate of {} != 16000, skipping.\n".format(filename))
            continue
        audio = np.mean(audio, axis=1)
        audio = np.expand_dims(audio, axis=1)
        audio_arr.append(audio)
    slices = [len(x) for x in audio_arr]
    audio_arr = np.concatenate(audio_arr)
    print("Num samples:")
    print("\ttotal: {}".format(sum(slices)))
    print("\tmin: {}".format(min(slices)))
    print("\tmax: {}".format(max(slices)))
    print("\tmean: {}".format(np.mean(slices)))
    print("\tstd: {}".format(np.std(slices)))

    names = [f.stem for f in filenames]
    write_netcdf_dataset(
        args.output, corpus=args.corpus, names=names, slices=slices,
        features=audio_arr, annotation_path=args.annotations
    )
    print("Wrote NetCDF4 dataset to {}.".format(args.output))


if __name__ == "__main__":
    main()
