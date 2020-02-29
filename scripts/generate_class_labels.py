import argparse
from pathlib import Path

from python.dataset import corpora

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('directory')


def main():
    args = parser.parse_args()

    with open(Path(args.corpus) / 'labels.txt', 'w') as fid:
        for p in sorted(Path(args.directory).glob('*.wav')):
            emotion = corpora[args.corpus].get_emotion(p.stem)
            print('{}, {}'.format(p.stem, emotion), file=fid)


if __name__ == "__main__":
    main()
