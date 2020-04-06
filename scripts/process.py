#!/usr/bin/python3

"""Batch process a list of files in a dataset using the openSMILE Toolkit."""

import argparse
import copy
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import arff
from tqdm import tqdm

from emotion_recognition.dataset import (corpora,
                                         parse_classification_annotations,
                                         parse_regression_annotations)

if sys.platform == 'win32':
    CONFIG_DIR = 'C:\\opensmile-2.3.0\\config'
    OPENSMILE_BIN = 'C:\\opensmile-2.3.0\\SMILExtract.exe'
else:
    CONFIG_DIR = '/usr/share/opensmile/config'
    OPENSMILE_BIN = '/usr/bin/SMILExtract'

parser = argparse.ArgumentParser(
    description="Processes audio data using openSMILE.")
required_args = parser.add_argument_group('Required args')
required_args.add_argument('--corpus', required=True,
                           help="Corpus to process",)
required_args.add_argument('--config', required=True,
                           help="Config file to use")
required_args.add_argument('--input_list',
                           help="File containing list of filenames")
required_args.add_argument('--annotations', help="Annotations file")


parser.add_argument('--debug', help="For debugging", default=False,
                    action='store_true')
parser.add_argument('--skip', help="Skip audio processing",
                    default=False, action='store_true')
parser.add_argument('--individual', help="Individual speaker output",
                    default=False, action='store_true')
parser.add_argument('--no_agg', help="Don't aggregate (save memory)",
                    default=False, action='store_true')

parser.add_argument('--out_dir', help="Directory to write data")
parser.add_argument('--prefix', help="Output file prefix")
parser.add_argument('--type', help="Type of annotations",
                    default='classification')
parser.add_argument('--filetype', default='arff',
                    help="Type of output file. One of {csv, arff}")


def read_arff(file: str):
    with open(file) as fid:
        return arff.load(fid)


def main():
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError("Config file doesn't exist")
    prefix = args.prefix or Path(args.config).stem

    input_file = args.input_list or Path(args.corpus) / 'classification.txt'
    with open(input_file) as fid:
        input_list = [x.strip() for x in fid.readlines()]
    names = [Path(f).stem for f in input_list]

    allowed_types = ['regression', 'classification']
    if args.type not in allowed_types:
        raise ValueError("--type must be one of", allowed_types)

    if args.type == 'regression':
        prefix += '_regression'

    tmp_dir = Path(args.corpus) / 'tmp'

    annotation_file = args.annotations or Path(args.corpus) / 'labels.csv'

    if not args.skip:
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor() as pool:
            futures = {}
            for filename in input_list:
                name = Path(filename).stem

                output_file = tmp_dir / '{}_{}.{}'.format(prefix, name,
                                                          args.filetype)
                if output_file.exists():
                    output_file.unlink()
                if args.filetype == 'arff':
                    outopt = '-output'
                elif args.filetype == 'csv':
                    outopt = '-csvoutput'
                else:
                    raise ValueError((
                        "--filetype must be on of {csv, arff}, got '{}'"
                        .format(args.filetype)
                    ))

                smile_args = [
                    OPENSMILE_BIN,
                    '-C', args.config,
                    '-I', filename,
                    outopt, output_file,
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

    if args.no_agg:
        return

    with ProcessPoolExecutor() as pool:
        futures = {}
        tmp_files = [
            tmp_dir / '{}_{}.arff'.format(prefix, name)
            for name in names
        ]
        tmp_files.sort()
        for filename in tmp_files:
            name = Path(filename).stem
            futures[name] = pool.submit(read_arff, filename)

        pbar = tqdm(futures.items(), desc="Processing data", unit='files')
        data_list = []
        for name, future in pbar:
            data_list.append(future.result())

    output_dir = (Path(args.out_dir) if args.out_dir
                  else Path(args.corpus) / 'output')
    output_dir.mkdir(parents=True, exist_ok=True)

    full_data = copy.deepcopy(data_list[0])
    full_data['relation'] = args.corpus
    full_data['data'] = []
    if args.type == 'regression':
        annotations = parse_regression_annotations(annotation_file)
        del full_data['attributes'][-1]
        keys = sorted(next(iter(annotations.values())).keys())
        full_data['attributes'].extend([('out_' + k, 'NUMERIC')
                                        for k in keys])
    elif args.type == 'classification':
        annotations = parse_classification_annotations(annotation_file)
        full_data['attributes'][-1] = ('emotion',
                                       sorted(set(annotations.values())))

    # This works for both utterance level and frame level features
    for data in data_list:
        name = data['data'][0][0]
        if args.type == 'regression':
            for i in range(len(data['data'])):
                del data['data'][i][-1]
                for _, v in sorted(annotations[name].items()):
                    data['data'][i].append(v)
        elif args.type == 'classification':
            for i in range(len(data['data'])):
                data['data'][i][-1] = annotations[name]
        full_data['data'].extend(data['data'])

    output_file = output_dir / '{}.arff'.format(prefix)
    with open(output_file, 'w') as fid:
        arff.dump(full_data, fid)
    print("Wrote {}".format(output_file))

    # This is mainly for WEKA since in our own scripts we can already
    # modify the datasets
    if args.individual:
        speakers = corpora[args.corpus].speakers
        speaker_dicts = [dict(data_list[0]) for _ in speakers]
        for d in speaker_dicts:
            d['data'] = []

        for inst in data_list:
            idx = speakers.index(
                corpora[args.corpus].get_speaker(inst['data'][0][0]))
            speaker_dicts[idx]['data'].append(inst['data'][0])

        for d, sp in zip(speaker_dicts, speakers):
            d['relation'] = args.corpus
            output_file = output_dir / '{}_{}.arff'.format(prefix, sp)
            with open(output_file, 'w') as fid:
                arff.dump(d, fid)
            print("Wrote {}".format(output_file))


if __name__ == "__main__":
    main()
