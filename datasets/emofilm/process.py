"""Process the raw EmoFilm dataset.

This assumes the file structure from the original compressed file:
/.../
    *.wav
"""

from pathlib import Path

import click
from emotion_recognition.dataset import (resample_audio, write_filelist,
                                         write_labels)
from emotion_recognition.utils import PathlibPath

emotion_map = {
    'ans': 'fear',
    'dis': 'disgust',
    'gio': 'happiness',
    'rab': 'anger',
    'tri': 'sadness'
}


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the EmoFilm dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob('*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob('*.wav'))
    write_labels({p.stem: emotion_map[p.stem[2:5]] for p in paths})


if __name__ == "__main__":
    main()
