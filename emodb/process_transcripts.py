#!/usr/bin/python3

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="EMO-DB transcriptions directory",
                    default='silb', type=str)
parser.add_argument('--wordlist', type=str,
                    help="File to write wordlist to.")


def main():
    args = parser.parse_args()

    utterances = {}
    for p in Path(args.dir).glob('*.silb'):
        with open(p, encoding='latin_1') as fid:
            words = []
            for line in fid:
                line = line.strip()
                time, word = line.split()
                if word in ['.', '(']:
                    continue
                words.append(word.strip())
            utterances[p.stem] = ' '.join(words)

    if args.wordlist:
        with open(args.wordlist, 'w') as fid:
            for u, s in sorted(utterances.items()):
                print('{}, {}'.format(u, s), file=fid)

        with open(Path(args.wordlist).with_suffix('.csv'), 'w') as fid:
            for u, s in sorted(utterances.items()):
                print('{};"{}"'.format(u, s), file=fid)


if __name__ == "__main__":
    main()
