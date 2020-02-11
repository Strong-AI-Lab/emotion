#!/usr/bin/python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

import arff
import tqdm

from python.dataset import read_annotations, corpora


class ProgressUpdater:
    """Class used for displaying a progress bar in terminal using tqdm.

    Args:
        total: the maximum value of the progress bar
        desc: a description to display in the progress bar
    """
    def __init__(self, total: int, desc: str, length=30, unit='files'):
        self.pbar = tqdm.tqdm(total=total, desc=desc, unit=unit)
        self.count = 0
        self.total = total
        self.length = length
        self.format = '>{}s'.format(self.length)
        self.lock = Lock()

    def __call__(self, name):
        with self.lock:
            if self.count < self.total:
                self.count += 1
                self.pbar.update(1)
                self.pbar.set_postfix_str(
                    format(name[-self.length:], self.format))
                if self.count == self.total:
                    self.pbar.close()


if sys.platform == 'win32':
    CONFIG_DIR = 'C:\\opensmile-2.3.0\\config'
    OPENSMILE_BIN = 'C:\\opensmile-2.3.0\\SMILExtract.exe'
else:
    CONFIG_DIR = '/usr/share/opensmile/config'
    OPENSMILE_BIN = '/usr/bin/SMILExtract'

configs = [f for f in os.listdir(CONFIG_DIR)]
configs += ['gemaps/' + f
            for f in os.listdir(os.path.join(CONFIG_DIR, 'gemaps'))]
configs = [c[:-5] for c in configs if c.endswith('.conf')]
configs = '\n'.join(sorted(configs))

DESCRIPTION = """Processes audio data using OpenSMILE.

Available configs:
""" + configs

parser = argparse.ArgumentParser(
    description=DESCRIPTION,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('corpus', help="Corpus to process")
parser.add_argument('config', help="Config file to use")

parser.add_argument('--debug', help="For debugging", default=False,
                    action='store_true')
parser.add_argument('-s', '--skip', help="Skip audio processing",
                    default=False, action='store_true')
parser.add_argument('-i', '--individual', help="Individual output",
                    default=False, action='store_true')

parser.add_argument('-a', '--annot_file', help="Annotations file")
parser.add_argument('-o', '--out_dir', help="Directory to write data")
parser.add_argument('-p', '--prefix', help="Output file prefix")
parser.add_argument('-t', '--type', help="Type of annotations",
                    default='classification')
parser.add_argument('-w', '--wav_dir',
                    help="Directory to search for WAV files")


def read_arff(file: str):
    with open(file) as fid:
        return arff.load(fid)


def main():
    args = parser.parse_args()

    conf_file = args.config + '.conf'
    if os.path.exists(args.config):
        config = args.config
    elif os.path.exists(conf_file):
        config = conf_file
    else:
        config = os.path.join(CONFIG_DIR, conf_file)
        if not os.path.exists(config):
            raise FileNotFoundError("Config file doesn't exist")

    output_prefix = args.prefix or args.config
    if output_prefix.startswith('gemaps/'):
        output_prefix = output_prefix[7:]

    allowed_types = ['regression', 'classification']
    if args.type not in allowed_types:
        raise ValueError("--type must be one of", allowed_types)

    if args.type == 'regression':
        output_prefix += '_regression'

    tmp_dir = os.path.join(args.corpus, 'tmp')

    if not args.skip:
        wav_dir = args.wav_dir or os.path.join(args.corpus, 'wav_corpus')
        if not os.path.exists(wav_dir):
            raise FileNotFoundError(
                "Directory does not exist: {}".format(wav_dir))
        os.makedirs(tmp_dir, exist_ok=True)

        with ProcessPoolExecutor() as pool:
            futures = {}
            wav_files = [f for f in os.listdir(wav_dir)
                         if f.endswith('.wav') or f.endswith('.WAV')]
            emotions = ','.join(corpora[args.corpus].emotion_map.values())
            for file in wav_files:
                name = os.path.splitext(file)[0]
                filepath = os.path.join(wav_dir, file)
                if args.type == 'classification':
                    emotion = corpora[args.corpus].get_emotion(name)
                    if emotion not in corpora[args.corpus].emotion_map:
                        continue
                    label = corpora[args.corpus].emotion_map[emotion]
                else:
                    emotions = 'unknown'
                    label = 'unknown'

                output_file = os.path.join(
                    tmp_dir, '{}_{}.arff'.format(output_prefix, name))
                if os.path.exists(output_file):
                    os.remove(output_file)

                smile_args = [
                    OPENSMILE_BIN,
                    '-C', config,
                    '-I', filepath,
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

            pbar = ProgressUpdater(len(wav_files), "Processing audio")
            try:
                for name, future in futures.items():
                    future.result()
                    pbar(name)
            except KeyboardInterrupt:
                print("Cancelling...")
                for future in futures.values():
                    future.cancel()
                sys.exit(1)

    with ProcessPoolExecutor() as pool:
        futures = {}
        tmp_files = [file for file in os.listdir(tmp_dir)
                     if file.startswith(output_prefix)]
        for file in tmp_files:
            name = os.path.splitext(file)[0]
            filepath = os.path.join(tmp_dir, file)
            futures[name] = pool.submit(read_arff, filepath)

        pbar = ProgressUpdater(len(tmp_files), "Processing data")
        data_list = []
        try:
            for name, future in sorted(futures.items()):
                data_list.append(future.result())
                pbar(name)
        except KeyboardInterrupt:
            print("Cancelling...")
            for future in futures.values():
                future.cancel()
            for future in futures.values():
                if future.running():
                    future.result()
            sys.exit(1)

    output_dir = args.out_dir or os.path.join(args.corpus, 'output')
    os.makedirs(output_dir, exist_ok=True)

    annot_file = args.annot_file or os.path.join(args.corpus, 'annot.txt')

    full_data = data_list[0]
    full_data['relation'] = args.corpus
    if args.type == 'regression':
        annotations = read_annotations(annot_file)
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
