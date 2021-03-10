"""Process the raw DEMoS dataset.

This assumes the file structure from the original compressed file:
/.../
    DEMOS/
        NP_*.wav
        PR_*.wav
    NEU/
        *.wav
    ...
"""

from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_annotations
from emorec.utils import PathlibPath

emotion_map = {
    'rab': 'anger',
    'tri': 'sadness',
    'gio': 'happiness',
    'pau': 'fear',
    'dis': 'disgust',
    'col': 'guilt',
    'sor': 'surprise'
}


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the DEMoS dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob('DEMOS/PR_*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist(resample_dir.glob('*.wav'))
    write_annotations({p.stem: emotion_map[p.stem[-6:-3]] for p in paths})
    write_annotations({p.stem: p.stem[-9:-7] for p in paths}, 'speaker')


if __name__ == "__main__":
    main()
