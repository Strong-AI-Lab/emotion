"""Process the raw CaFE dataset.

This assumes the file structure from the original compressed file:
/.../
    Colère/
        Faible/
            *.aiff [for 192k data]
            *.wav  [for 48k data]
        Fort/
    ...
"""

from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_labels
from emorec.utils import PathlibPath

emotion_map = {
    'C': 'anger',
    'D': 'disgust',
    'J': 'happiness',
    'N': 'neutral',
    'P': 'fear',
    'S': 'surprise',
    'T': 'sadness'
}


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the CaFE dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob('**/*.wav'))
    if len(paths) == 0:
        paths = list(input_dir.glob('**/*.aiff'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob('*.wav'))
    write_labels({p.stem: emotion_map[p.stem[3]] for p in paths})


if __name__ == "__main__":
    main()
