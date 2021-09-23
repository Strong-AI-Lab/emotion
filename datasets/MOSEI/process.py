"""Process the raw CMU-MOSEI dataset.

This assumes the file structure from the original compressed file:
/.../
    CMU_MOSEI_Labels.csd
    *.csd
    Raw/
        Audio/
            Full/
                WAV_16000/
                    *.wav
                COVAREP/
                    *.mat
        Videos/
            Full/
                Combined/
                    *.mp4
            Segmented/
                Combined/
                    *.mp4
    ...
"""

import json
from itertools import chain
from pathlib import Path

import click
import h5py
import numpy as np
import soundfile
from joblib import delayed

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import PathlibPath, TqdmParallel


@click.command()
@click.argument("input_dir", type=PathlibPath(exists=True, file_okay=False))
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CMU-MOSEI dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    audio_dir = input_dir / "Raw" / "Audio" / "Full" / "WAV_16000"
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(parents=True, exist_ok=True)

    dataset = h5py.File(input_dir / "CMU_MOSEI_Labels.csd", "r")
    dim_names = json.loads(dataset["All Labels/metadata/dimension names"][0])
    names = list(dataset["All Labels/data"].keys())
    dataset.close()

    def process(name: str):
        dataset = h5py.File(input_dir / "CMU_MOSEI_Labels.csd", "r")
        features = dataset[f"All Labels/data/{name}/features"]
        intervals = np.array(dataset[f"All Labels/data/{name}/intervals"])
        intervals = (16000 * intervals).astype(int)
        with open(audio_dir / (name + ".wav"), "rb") as fid:
            audio, _ = soundfile.read(fid)
        labels = []
        for i in range(len(intervals)):
            newname = f"{name}_{i:02d}"
            segment = audio[slice(*intervals[i])]
            with open(resample_dir / (newname + ".wav"), "wb") as fid:
                soundfile.write(fid, segment, 16000)
            labels.append((newname, dim_names[1:][features[i][1:].argmax()]))
        return labels

    lab_lists = TqdmParallel(total=len(names), desc="Processing files", n_jobs=-1)(
        delayed(process)(name) for name in names
    )
    label_dict = dict(chain.from_iterable(lab_lists))
    write_annotations(label_dict, "label")
    write_filelist(resample_dir.glob("*.wav"), "files_all")


if __name__ == "__main__":
    main()
