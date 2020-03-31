import argparse
from pathlib import Path

from emotion_recognition.dataset import corpora

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', required=True)
parser.add_argument('--directory', required=True)
parser.add_argument('--output', required=True)


def main():
    args = parser.parse_args()

    with open(args.output, 'w') as fid:
        print("Name,Emotion", file=fid)
        for p in sorted(Path(args.directory).glob('*.wav')):
            emotion = corpora[args.corpus].get_emotion(p.stem)
            try:
                emotion = corpora[args.corpus].emotion_map[emotion]
            except KeyError:
                emotion = 'unknown'
            print('{},{}'.format(p.stem, emotion), file=fid)


if __name__ == "__main__":
    main()
