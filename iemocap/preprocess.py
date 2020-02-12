#!/usr/bin/python3

import argparse
import glob
import os
import re
import shutil

# [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
REGEX = (r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t'
         r'\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="IEMOCAP annotations directory",
                    default='annotations', type=str)
parser.add_argument('--annotations', help="File to write annotations to",
                    default='annot.txt', type=str)
parser.add_argument('--wav_in', help="Directory storing WAV files",
                    default='sentences', type=str)
parser.add_argument('--wav_out', help="Directory to output renamed files",
                    default='wav_corpus', type=str)


def main():
    args = parser.parse_args()

    regex = re.compile(REGEX)
    dimensions = {}
    labels = {}
    for filename in glob.iglob(os.path.join(args.dir, '*.txt')):
        with open(filename) as fid:
            for line in fid:
                line = line.strip()
                match = regex.match(line)
                if match:
                    dimensions[match.group(3)] = (match.group(5),
                                                  match.group(6),
                                                  match.group(7))
                    labels[match.group(3)] = match.group(4)

    if args.annotations:
        with open(args.annotations, 'w') as fid:
            for name, (v, a, d) in dimensions.items():
                print('{}, V: {}, A: {}, D: {}'.format(name, v, a, d),
                      file=fid)

    if args.wav_in and args.wav_out:
        for name, emo in labels.items():
            src = os.path.join(args.wav_in, '{}.wav'.format(name))
            dst = os.path.join(args.wav_out, '{}_{}.wav'.format(name, emo))
            shutil.copy(src, dst)


if __name__ == "__main__":
    main()
