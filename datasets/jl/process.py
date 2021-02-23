"""Process the raw JL dataset.

This assumes the file structure from the original compressed file:
/.../
    Raw JL Corpus (unchecked ...)/
        JL(wav+txt)/
            *.wav
        ...
    ...
"""

import re
from pathlib import Path

import click
from emorec.dataset import resample_audio, write_filelist, write_labels
from emorec.utils import PathlibPath

REGEX = re.compile(r'^(?:fe)?male[12]_([a-z]+)_\d+[ab]_[12]$')

emotion_map = {
    'angry': 'anger',
    'sad': 'sadness',
    'neutral': 'neutral',
    'happy': 'happiness',
    'excited': 'excitedness',
    'anxious': 'anxiety',
    'apologetic': 'apologetic',
    'assertive': 'assertive',
    'concerned': 'concern',
    'encouraging': 'encouraging'
}

unused_emotions = ['anxious', 'apologetic', 'assertive', 'concerned',
                   'encouraging']


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
def main(input_dir: Path):
    """Process the JL dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob(
        'Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav'))
    resample_dir = Path('resampled')
    resample_audio(paths, resample_dir)

    write_filelist([p for p in resample_dir.glob('*.wav')
                    if REGEX.match(p.stem).group(1) not in unused_emotions])
    write_labels({p.stem: emotion_map.get(REGEX.match(p.stem).group(1))
                  for p in paths})


if __name__ == "__main__":
    main()
