#!/usr/bin/python3

import argparse
import os
import re
import shutil

REGEX = (r'^$')

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="SmartKom annotations file",
                    default='Evalution.txt', type=str)
parser.add_argument('--annotations', help="File to write annotations to",
                    default='annot.txt', type=str)
parser.add_argument('--wav_in', help="Directory storing WAV files",
                    default='all', type=str)
parser.add_argument('--wav_out', help="Directory to output renamed files",
                    default='wav_corpus', type=str)


def main():
    args = parser.parse_args()

    regex = re.compile(REGEX)
    dimensions = {}
    labels = {}
    with open(args.dir) as fid:
        for line in fid:
            line = line.strip()
            match = regex.match(line)
            if match:
                dimensions[match.group(1)] = [match.group(i)
                                              for i in [3, 4, 5, 6]]
                labels[match.group(1)] = match.group(2)

    with open(args.annotations, 'w') as fid:
        for name, (a, v, d, n) in dimensions.items():
            print('{}, A: {}, V: {}, D: {}, N: {}'.format(name, a, v, d, n),
                  file=fid)

    if args.wav_in and args.wav_out:
        for name, emo in labels.items():
            src = os.path.join(args.wav_in, '{}.wav'.format(name))
            dst = os.path.join(args.wav_out, '{}-{}.wav'.format(name, emo))
            shutil.copy(src, dst)


if __name__ == "__main__":
    main()
