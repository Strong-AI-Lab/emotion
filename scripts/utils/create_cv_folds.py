#!/usr/bin/python

"""Create directory structure with cross-validation folds."""

import argparse
import shutil
from pathlib import Path

from emotion_recognition.dataset import (corpora,
                                         parse_classification_annotations)

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', required=True, type=str)
parser.add_argument('--output', help="Directory in which to place folds.",
                    required=True, type=Path)
parser.add_argument('--input_list', required=True, type=Path,
                    help="File containing list of filenames")
parser.add_argument('--annotations', help="Annotations file", type=Path)


def main():
    args = parser.parse_args()

    annotations = parse_classification_annotations(args.annotations)

    get_speaker = corpora[args.corpus].get_speaker
    speakers = {s: [] for s in corpora[args.corpus].speakers}
    with open(args.input_list) as fid:
        for line in fid:
            filepath = Path(line.strip())
            speaker = get_speaker(filepath.stem)
            speakers[speaker].append(filepath)

    for i, speaker in enumerate(speakers.keys()):
        for path in speakers[speaker]:
            emotion = annotations[path.stem]
            fold = 'fold_{:d}'.format(i + 1)
            newpath = args.output / fold / emotion / path.name
            newpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(newpath))
            print(newpath)


if __name__ == "__main__":
    main()
