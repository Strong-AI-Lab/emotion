#!/usr/bin/python3

"""Batch process a list of files in a dataset using the openSMILE Toolkit."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from emotion_recognition.dataset import write_netcdf_dataset
from joblib import Parallel, delayed

if sys.platform == 'win32':
    OPENSMILE_BIN = 'third_party\\opensmile\\SMILExtract.exe'
else:
    OPENSMILE_BIN = 'third_party/opensmile/SMILExtract'


def opensmile(path: Union[str, Path], config: Union[str, Path],
              tmp: Union[str, Path] = 'tmp', debug: bool = False,
              restargs: List[str] = []):
    name = Path(path).stem
    output_file = Path(tmp) / '{}.csv'.format(name)
    if output_file.exists():
        output_file.unlink()

    smile_args = [
        OPENSMILE_BIN,
        '-C', str(config),
        '-I', str(path),
        '-csvoutput', str(output_file),
        '-classes', '{unknown}',
        '-class', 'unknown',
        '-instname', name,
        *restargs
    ]

    stdout = subprocess.DEVNULL
    if debug:
        stdout = None
    subprocess.run(smile_args, stdout=stdout, stderr=subprocess.STDOUT)


def process_csv(path: Union[str, Path]):
    df = pd.read_csv(path, quotechar="'", header=None)
    return df.iloc[:, 1:]


def main():
    parser = argparse.ArgumentParser(
        description="Processes audio data using openSMILE.")
    # Required args
    required_args = parser.add_argument_group('Required args')
    required_args.add_argument('--corpus', type=str, required=True,
                               help="Corpus to process")
    required_args.add_argument('--config', type=Path, required=True,
                               help="Config file to use")
    required_args.add_argument('--input', type=Path, required=True,
                               help="File containing list of files")
    required_args.add_argument('--output', type=Path, required=True,
                               help="Output file.")

    # Flags
    parser.add_argument('--debug', action='store_true',
                        help="For debugging")

    # Optional args
    parser.add_argument('--type', default='classification',
                        help="Type of annotations")
    parser.add_argument('--annotations', type=Path, help="Annotations file")

    args, restargs = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError("Config file doesn't exist")

    with open(args.input) as fid:
        input_list = [x.strip() for x in fid.readlines()]
    names = sorted(Path(f).stem for f in input_list)

    allowed_types = ['regression', 'classification']
    if args.type not in allowed_types:
        raise ValueError("--type must be one of", allowed_types)

    with tempfile.TemporaryDirectory(prefix='opensmile',
                                     suffix=args.corpus) as tmp:
        parallel_args = dict(n_jobs=1 if args.debug else -1, verbose=1)
        Parallel(prefer='threads', **parallel_args)(
            delayed(opensmile)(path, args.config, tmp, args.debug, restargs)
            for path in input_list
        )

        tmp_files = [Path(tmp) / '{}.csv'.format(name) for name in names]
        missing = [f for f in tmp_files if not f.exists()]
        if len(missing) > 0:
            msg = "Not all audio files were processed properly. These files " \
                  "are missing:\n" + '\n'.join(map(str, missing))
            raise RuntimeError(msg)
        # Use CPUs for this because I don't think it releases the GIL
        # for the whole processing.
        arr_list = Parallel(**parallel_args)(
            delayed(process_csv)(path) for path in tmp_files
        )

    # This should be a 2D array
    full_array = np.concatenate(arr_list, axis=0)
    assert len(full_array.shape) == 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(
        args.output, corpus=args.corpus, names=names, features=full_array,
        slices=[x.shape[0] for x in arr_list],
        annotation_path=args.annotations, annotation_type=args.type
    )

    print("Wrote netCDF dataset to {}".format(args.output))


if __name__ == "__main__":
    main()
