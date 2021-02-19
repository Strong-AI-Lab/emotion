"""Process the raw Portuguese dataset (Castro & Lima).

This assumes the file structure from the original compressed file:
/.../
    *.wav
"""

import re
from pathlib import Path

import click
from emorec.dataset import (resample_audio, write_filelist,
                                         write_labels)
from emorec.utils import PathlibPath

REGEX = re.compile(r'^\d+[sp][AB]_([a-z]+)\d+$')

emotion_map = {
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happiness',
    'sad': 'sadness',
    'neutral': 'neutral',
    'surprise': 'surprise'
}


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the Portuguese dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob('*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob('*.wav'))
    write_labels({p.stem: emotion_map[REGEX.match(p.stem).group(1)]
                  for p in paths})


if __name__ == "__main__":
    main()
