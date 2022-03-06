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
from collections import defaultdict
from pathlib import Path
from typing import Dict

import click
import h5py
import pandas as pd
import soundfile
from joblib import delayed
from tqdm import tqdm

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import TqdmParallel


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CMU-MOSEI dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    audio_dir = input_dir / "Raw" / "Audio" / "Full" / "WAV_16000"
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True)
    all_videos = sorted(set(x.stem for x in audio_dir.glob("*.wav")))

    transcripts_dir = input_dir / "Raw" / "Transcript" / "Segmented" / "Combined"
    transcripts = []
    for name in all_videos:
        trn_file = transcripts_dir / f"{name}.txt"
        with open(trn_file) as fid:
            for row in fid:
                # Some transcripts contain '_____' or '______'
                vals = row.strip().split("___", maxsplit=5)
                dtypes = [str, str, float, float, str]
                transcripts.append([f(x) for f, x in zip(dtypes, vals)])
    trn_df = pd.DataFrame(transcripts, columns=["video", "clip", "start", "end", "trn"])
    trn_df.index = trn_df["video"] + "_" + trn_df["clip"]
    write_annotations(trn_df["trn"].to_dict(), "transcript")

    def process(seg):
        start = int(16000 * trn_df.loc[seg, "start"])
        end = int(16000 * trn_df.loc[seg, "end"])
        base_clip = audio_dir / f"{trn_df.loc[seg, 'video']}.wav"
        audio, _ = soundfile.read(base_clip, start=start, stop=end)
        soundfile.write(resample_dir / f"{seg}.wav", audio, samplerate=16000)

    if resample:
        TqdmParallel(len(trn_df.index), "Splitting audio", prefer="threads", n_jobs=-1)(
            delayed(process)(name) for name in trn_df.index
        )

    dataset = h5py.File(input_dir / "CMU_MOSEI_Labels.csd", "r")
    dim_names = json.loads(dataset["All Labels/metadata/dimension names"][0])
    labelled_videos = set(dataset["All Labels/data"].keys())

    labels = {}
    dim_vals: Dict[str, Dict[str, float]] = defaultdict(dict)
    for name in tqdm(trn_df.index, "Processing labels"):
        video = trn_df.loc[name, "video"]
        if video in labelled_videos:
            features = dataset[f"All Labels/data/{video}/features"]
            intervals = dataset[f"All Labels/data/{video}/intervals"]
            for i, (start, end) in enumerate(intervals):
                if (
                    start == trn_df.loc[name, "start"]
                    and end == trn_df.loc[name, "end"]
                ):
                    labels[name] = dim_names[1:][features[i][1:].argmax()]
                    for d_idx, dim in enumerate(dim_names):
                        dim_vals[dim][name] = features[i][d_idx]
                    break
        else:
            labels[name] = ""
            for dim in dim_names:
                dim_vals[dim][name] = float("nan")
    dataset.close()

    write_filelist(resample_dir.glob("*.wav"), "files_all")
    write_annotations(labels, "label")
    for dim in dim_names:
        write_annotations(dim_vals[dim], dim)
    labelled_paths = [resample_dir / f"{name}.wav" for name in labels]
    write_filelist(labelled_paths, "files_labels")


if __name__ == "__main__":
    main()
