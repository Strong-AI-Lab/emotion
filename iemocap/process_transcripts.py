#!/usr/bin/python3

import argparse
import re
from pathlib import Path

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

REGEX = r'^(Ses0[1-5][MF]_(?:impro|script)0[1-9][ab]?(?:_\db?)?_[MF][X\d]{2}\d) \[\d+\.\d+-\d+\.\d+\]:(.*)$'  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('dir', help="IEMOCAP transcriptions directory",
                    default='transcriptions', type=str)
parser.add_argument('--wordlist', type=str,
                    help="File to write wordlist to.")

stops = set(stopwords.words('english'))


def clean(words):
    words = [x for x in words.split() if '[' not in x and ']' not in x]
    return ' '.join(words)


def main():
    args = parser.parse_args()

    regex = re.compile(REGEX)

    utterances = {}
    for p in Path(args.dir).glob('*.txt'):
        with open(p) as fid:
            for line in fid:
                line = line.strip()
                match = regex.match(line)
                if match:
                    utterances[match.group(1)] = match.group(2).strip()
    utterances = {u: clean(x) for u, x in utterances.items()}

    if args.wordlist:
        with open(args.wordlist, 'w') as fid:
            for u, s in sorted(utterances.items()):
                print('{}, {}'.format(u, s), file=fid)

        with open(Path(args.wordlist).with_suffix('.csv'), 'w') as fid:
            for u, s in sorted(utterances.items()):
                tokens = word_tokenize(s)
                tokens = [x for x in tokens if x not in stops]
                tokens = [x for x in tokens if not re.search(r'[.!$,;:?]', x)]
                words = [w.lower() for w in tokens]
                s = ' '.join(words)
                s = s.replace('"', r'\"')
                print('{};"{}"'.format(u, s), file=fid)


if __name__ == "__main__":
    main()
