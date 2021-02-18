"""Process the raw ShEMO dataset.

This assumes the file structure from the original compressed file:
/.../
    male/
        *.wav
    female/
    ...
"""

from pathlib import Path

import click
from emotion_recognition.dataset import (resample_audio, write_filelist,
                                         write_labels)
from emotion_recognition.utils import PathlibPath

emotion_map = {
    'A': 'anger',
    'H': 'happiness',
    'N': 'neutral',
    'S': 'sadness',
    'W': 'surprise',
    'F': 'fear'
}

unused_emotions = ['F']


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the ShEMO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob('*/*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist([p for p in resample_dir.glob('*.wav')
                    if p.stem[3] not in unused_emotions])
    write_labels({p.stem: emotion_map[p.stem[3]] for p in paths})


if __name__ == "__main__":
    main()