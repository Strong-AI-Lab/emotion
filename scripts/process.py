#!/usr/bin/python3

"""Batch process a list of files in a dataset using the Opensmile Toolkit."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import arff
from tqdm import tqdm

from python.dataset import read_annotations, corpora


if sys.platform == 'win32':
    CONFIG_DIR = 'C:\\opensmile-2.3.0\\config'
    OPENSMILE_BIN = 'C:\\opensmile-2.3.0\\SMILExtract.exe'
else:
    CONFIG_DIR = '/usr/share/opensmile/config'
    OPENSMILE_BIN = '/usr/bin/SMILExtract'

parser = argparse.ArgumentParser(
    description="Processes audio data using OpenSMILE.")
required_args = parser.add_argument_group('Required args')
required_args.add_argument('--corpus', help="Corpus to process",
                           required=True)
required_args.add_argument('--config', help="Config file to use",
                           required=True)
required_args.add_argument(
    '--input_list', help="File containing list of filenames", required=True)


parser.add_argument('--debug', help="For debugging", default=False,
                    action='store_true')
parser.add_argument('--skip', help="Skip audio processing",
                    default=False, action='store_true')
parser.add_argument('--individual', help="Individual output",
                    default=False, action='store_true')

parser.add_argument('--annotations', help="Annotations file")
parser.add_argument('--out_dir', help="Directory to write data")
parser.add_argument('--prefix', help="Output file prefix")
parser.add_argument('--type', help="Type of annotations",
                    default='classification')


def read_arff(file: str):
    with open(file) as fid:
        return arff.load(fid)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("Config file doesn't exist")
    config = os.path.splitext(os.path.basename(args.config))[0]
    output_prefix = args.prefix or config

    with open(args.input_list) as fid:
        input_list = [x.strip() for x in fid.readlines()]
    names = [os.path.splitext(os.path.basename(f))[0] for f in input_list]

    allowed_types = ['regression', 'classification']
    if args.type not in allowed_types:
        raise ValueError("--type must be one of", allowed_types)

    if args.type == 'regression':
        output_prefix += '_regression'

    tmp_dir = os.path.join(args.corpus, 'tmp')

    if not args.skip:
        os.makedirs(tmp_dir, exist_ok=True)

        with ProcessPoolExecutor() as pool:
            emotions = 'unknown'
            if args.type == 'classification':
                emotions = ','.join(corpora[args.corpus].emotion_map.values())
            label = 'unknown'

            futures = {}
            for filename in input_list:
                name = os.path.splitext(os.path.basename(filename))[0]
                if args.type == 'classification':
                    emotion = corpora[args.corpus].get_emotion(name)
                    if emotion not in corpora[args.corpus].emotion_map:
                        continue
                    label = corpora[args.corpus].emotion_map[emotion]

                output_file = os.path.join(
                    tmp_dir, '{}_{}.arff'.format(output_prefix, name))
                if os.path.exists(output_file):
                    os.remove(output_file)

                smile_args = [
                    OPENSMILE_BIN,
                    '-C', args.config,
                    '-I', filename,
                    '-O', output_file,
                    '-classes', '{{{}}}'.format(emotions),
                    '-class', label,
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

    with ProcessPoolExecutor() as pool:
        futures = {}
        tmp_files = [
            os.path.join(tmp_dir, '{}_{}.arff'.format(output_prefix, name))
            for name in names
        ]
        tmp_files.sort()
        for filename in tmp_files:
            name = os.path.splitext(os.path.basename(filename))[0]
            futures[name] = pool.submit(read_arff, filename)

        pbar = tqdm(futures.items(), desc="Processing data", unit='files')
        data_list = []
        for name, future in pbar:
            data_list.append(future.result())

    output_dir = args.out_dir or os.path.join(args.corpus, 'output')
    os.makedirs(output_dir, exist_ok=True)

    full_data = data_list[0]
    full_data['relation'] = args.corpus
    if args.type == 'regression':
        annotations = read_annotations(args.annotations)
        del full_data['attributes'][-1]
        keys = sorted(next(iter(annotations.values())).keys())
        full_data['attributes'].extend(
            [('regression_' + k, 'NUMERIC') for k in keys])

    # This works for both utterance level and frame level features
    for data in data_list[1:]:
        if args.type == 'regression':
            name = data['data'][0][0]
            for i in range(len(data['data'])):
                del data['data'][i][-1]
                for _, v in sorted(annotations[name].items()):
                    data['data'][i].append(v)
        full_data['data'].extend(data['data'])

    output_file = os.path.join(
        output_dir, '{}.arff'.format(output_prefix))
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
            output_file = os.path.join(
                output_dir, '{}_{}.arff'.format(output_prefix, sp))
            with open(output_file, 'w') as fid:
                arff.dump(d, fid)
            print("Wrote {}".format(output_file))


if __name__ == "__main__":
    main()
