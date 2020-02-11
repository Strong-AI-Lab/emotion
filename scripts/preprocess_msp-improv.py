#!/usr/bin/python3

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('annotations', help="MSP-IMPROV annotations file", default='msp-improv/Evalution.txt', type=str)
parser.add_argument('outfile', help="File to write annotations to", default='msp-improv/annot.txt', type=str)


def main():
    args = parser.parse_args()

    regex = re.compile(r'^UTD-IMPROV-([A-Z0-9-]+)\.avi; [A-Z]; A:(\d+\.\d+); V:(\d+\.\d+); D:(\d+\.\d+) ; N:(\d+\.\d+);$')
    dimensions = {}
    with open(args.annotations) as fid:
        for line in fid:
            line = line.strip()
            match = regex.match(line)
            if match:
                dimensions[match.group(1)] = [match.group(i) for i in [2, 3, 4, 5]]

    with open(args.outfile, 'w') as fid:
        for name, (a, v, d, n) in dimensions.items():
            print('{}, A: {}, V: {}, D: {}, N: {}'.format(name, a, v, d, n), file=fid)


if __name__ == "__main__":
    main()
