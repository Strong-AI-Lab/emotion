#!/usr/bin/python3

"""Batch process a list of files in a dataset using the openSMILE Toolkit."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import arff
import netCDF4
import numpy as np
import pandas as pd
from tqdm import tqdm

from emotion_recognition.dataset import (parse_classification_annotations,
                                         parse_regression_annotations)

if sys.platform == 'win32':
    OPENSMILE_BIN = 'third_party\\opensmile\\SMILExtract.exe'
else:
    OPENSMILE_BIN = 'third_party/opensmile/SMILExtract'


def read_arff(file: str):
    with open(file) as fid:
        return arff.load(fid)


def main():
    parser = argparse.ArgumentParser(
        description="Processes audio data using openSMILE.")
    # Required args
    required_args = parser.add_argument_group('Required args')
    required_args.add_argument('--corpus', required=True,
                               help="Corpus to process",)
    required_args.add_argument('--config', type=Path, required=True,
                               help="Config file to use")
    required_args.add_argument('--annotations', type=Path, required=True,
                               help="Annotations file")
    required_args.add_argument('--input_list', type=Path, required=True,
                               help="File containing list of filenames")

    # Flags
    parser.add_argument('--debug', help="For debugging", default=False,
                        action='store_true')
    parser.add_argument('--skip', help="Skip audio processing",
                        default=False, action='store_true')
    parser.add_argument('--noagg', help="Don't aggregate (save memory)",
                        default=False, action='store_true')

    # Optional args
    parser.add_argument('--output', type=Path, help="Directory to write data")
    parser.add_argument('--prefix', help="Output file prefix")
    parser.add_argument('--type', help="Type of annotations",
                        default='classification')

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError("Config file doesn't exist")
    prefix = args.prefix or args.config.stem

    with open(args.input_list) as fid:
        input_list = [x.strip() for x in fid.readlines()]
    names = sorted(Path(f).stem for f in input_list)

    allowed_types = ['regression', 'classification']
    if args.type not in allowed_types:
        raise ValueError("--type must be one of", allowed_types)

    if args.type == 'regression':
        prefix += '_regression'

    tmp_dir = Path('tmp') / args.corpus

    if not args.skip:
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor() as pool:
            futures = {}
            for filename in input_list:
                name = Path(filename).stem

                output_file = tmp_dir / '{}_{}.csv'.format(prefix, name)
                if output_file.exists():
                    output_file.unlink()

                smile_args = [
                    OPENSMILE_BIN,
                    '-C', str(args.config),
                    '-I', filename,
                    '-csvoutput', output_file,
                    '-classes', '{unknown}',
                    '-class', 'unknown',
                    '-instname', name
                ]

                stdout = subprocess.DEVNULL
                if args.debug:
                    stdout = None
                futures[name] = pool.submit(
                    subprocess.run,
                    smile_args,
                    stdout=stdout,
                    stderr=stdout)

            pbar = tqdm(futures.items(), desc="Processing audio", unit='files')
            try:
                for name, future in pbar:
                    future.result()
            except KeyboardInterrupt:
                print("Cancelling...")
                for future in futures.values():
                    future.cancel()
                pbar.close()
                sys.exit(1)

    if args.noagg:
        return

    with ProcessPoolExecutor() as pool:
        futures = {}
        tmp_files = (tmp_dir / '{}_{}.csv'.format(prefix, name)
                     for name in names)
        for filename in tmp_files:
            name = filename.stem
            futures[name] = pool.submit(
                partial(pd.read_csv, quotechar="'", header=None), filename)

        pbar = tqdm(futures.items(), desc="Processing data", unit='files')
        arr_list = []
        for name, future in pbar:
            df = future.result()
            arr_list.append(df.iloc[:, 1:])  # Ignore name

    # This should be a 2D array
    full_array = np.concatenate(arr_list, axis=0)
    assert len(full_array.shape) == 2

    if args.type == 'regression':
        annotations = parse_regression_annotations(args.annotations)
    else:
        annotations = parse_classification_annotations(args.annotations)

    output_dir = args.output or Path('output') / args.corpus
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '{}.nc'.format(prefix)

    dataset = netCDF4.Dataset(output_file, 'w')
    dataset.createDimension('instance', len(arr_list))
    dataset.createDimension('concat', full_array.shape[0])
    dataset.createDimension('features', full_array.shape[1])

    # This is so we can have both vectors and vlen sequences in the same
    # format. We just need to manipulate the array when we read in the dataset.
    slices = dataset.createVariable('slices', int, ('instance',))
    slices[:] = [x.shape[0] for x in arr_list]

    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = np.array(names)

    if args.type == 'regression':
        keys = next(iter(annotations.values())).keys()
        for k in keys:
            var = dataset.createVariable(k, np.float32, ('instance',))
            var[:] = np.array([annotations[x][k] for x in names])
        dataset.setncattr_string(
            'annotation_vars', json.dumps([k for k in keys]))
    else:
        label_nominal = dataset.createVariable('label_nominal', str,
                                               ('instance',))
        label_nominal[:] = np.array([annotations[x] for x in names])
        dataset.setncattr_string(
            'annotation_vars', json.dumps(['label_nominal']))

    features = dataset.createVariable('features', np.float32,
                                      ('concat', 'features'))
    features[:, :] = full_array

    dataset.setncattr_string('feature_dims',
                             json.dumps(['concat', 'features']))
    dataset.setncattr_string('corpus', args.corpus or '')
    dataset.close()

    print("Wrote {}".format(output_file))


if __name__ == "__main__":
    main()
