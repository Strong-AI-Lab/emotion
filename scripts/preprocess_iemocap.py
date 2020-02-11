#!/usr/bin/python3

import argparse
import glob
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="IEMOCAP annotations directory", default='iemocap/annotations', type=str)
parser.add_argument('outfile', help="File to write annotations to", default='iemocap/annot.txt', type=str)


def main():
    args = parser.parse_args()
    annotations_dir = args.dir

    regex = re.compile(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')
    utterances = {}
    for filename in glob.iglob(os.path.join(annotations_dir, '*.txt')):
        with open(filename) as fid:
            for line in fid:
                line = line.strip()
                match = regex.match(line)
                if match:
                    utterances[match.group(3)] = (match.group(5), match.group(6), match.group(7))

    with open(args.outfile, 'w') as fid:
        for name, (v, a, d) in utterances.items():
            print('{}, V: {}, A: {}, D: {}'.format(name, v, a, d), file=fid)


if __name__ == "__main__":
    main()
