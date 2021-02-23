"""Process the raw RAVDESS dataset.

This assumes the file structure from the original compressed file:
/.../
    Audio/
        Actor_01/
            *.wav
        ...
    ...
"""

from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_labels
from emorec.utils import PathlibPath

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happiness',
    '04': 'sadness',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the RAVDESS dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob('Audio/Actor_??/03-01-*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob('*.wav'))
    write_labels({p.stem: emotion_map[p.stem[6:8]] for p in paths})


if __name__ == "__main__":
    main()
