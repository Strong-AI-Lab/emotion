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

import click
import h5py
import numpy as np
import pandas as pd
import soundfile
from joblib import delayed
from splits import standard_test_fold, standard_train_fold, standard_valid_fold
from tqdm import tqdm

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import TqdmParallel

emotion_map = {
    "happy": "happiness",
    "sad": "sadness",
}


def get_vote(vals):
    idx = vals.argmax()
    max = vals[idx]
    if max > vals.sum() / 2:
        # Majority
        return idx, "M"
    if np.count_nonzero(max == vals) == 1:
        # Plurality
        return idx, "P"
    return -1, "X"


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CMU-MOSEI dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit FLAC audio.
    """

    audio_dir = input_dir / "Raw" / "Audio" / "Full" / "WAV_16000"
    all_videos = sorted({x.stem for x in audio_dir.glob("*.wav")})

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
    neg_starts = trn_df[trn_df["start"] < 0]
    if len(neg_starts) > 0:
        print(f"WARNING: {len(neg_starts)} clips with negative start value")
    trn_df.index = trn_df["video"] + "_" + trn_df["clip"]
    write_annotations(trn_df["trn"].to_dict(), "transcript")
    write_annotations(trn_df["video"].to_dict(), "video")

    dfs = []
    for csvfile in (input_dir / "Raw" / "Labels").glob("*.csv"):
        dfs.append(pd.read_csv(csvfile, dtype={15: str, 27: str, 28: str}))
    df = pd.concat(dfs)
    df["name"] = df["Input.VIDEO_ID"].str.cat(df["Input.CLIP"], sep="_")
    df["name"] = df["name"].map(lambda x: x.split("/")[1] if "/" in x else x)
    df["rater"] = df["WorkerId"]
    df.set_index(["name", "rater"], inplace=True)
    df.rename(
        columns={x: x[7:] for x in df.columns if x.startswith("Answer.")}, inplace=True
    )
    df = df[
        ["sentiment", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    ]
    df.sort_index().to_csv("ratings.csv")

    def process(seg):
        start = max(0, int(16000 * trn_df.loc[seg, "start"]))
        end = max(0, int(16000 * trn_df.loc[seg, "end"]))
        if end == start:
            return
        base_clip = audio_dir / f"{trn_df.loc[seg, 'video']}.wav"
        audio, _ = soundfile.read(base_clip, start=start, stop=end)
        soundfile.write(
            resample_dir / f"{seg}.flac", audio, samplerate=16000, subtype="PCM_16"
        )

    resample_dir = Path("resampled")
    resample_dir.mkdir(exist_ok=True)
    if resample:
        TqdmParallel(len(trn_df.index), "Splitting audio", prefer="threads", n_jobs=-1)(
            delayed(process)(name) for name in trn_df.index
        )
    all_paths = list(resample_dir.glob("*.flac"))
    write_filelist(all_paths, "files_all")
    write_annotations({x.stem: "en" for x in all_paths}, "language")

    dataset = h5py.File(input_dir / "CMU_MOSEI_Labels.csd", "r")
    dim_names = json.loads(dataset["All Labels/metadata/dimension names"][0])
    labelled_videos = set(dataset["All Labels/data"].keys())

    labels_maj = {}
    labels_plu = {}
    dim_vals: dict[str, dict[str, float]] = defaultdict(dict)
    for name in tqdm(trn_df.index, "Processing labels"):
        video = trn_df.loc[name, "video"]
        if video in labelled_videos:
            features = dataset[f"All Labels/data/{video}/features"]
            intervals = dataset[f"All Labels/data/{video}/intervals"]
            # We need to match the clip IDs to the intervals
            for i, (start, end) in enumerate(intervals):
                if (
                    start == trn_df.loc[name, "start"]
                    and end == trn_df.loc[name, "end"]
                ):
                    max_emo_idx, tp = get_vote(features[i, 1:])
                    labels_maj[name] = (
                        dim_names[1:][max_emo_idx] if tp == "M" else "unknown"
                    )
                    labels_plu[name] = (
                        dim_names[1:][max_emo_idx] if tp in ["M", "P"] else "unknown"
                    )
                    for d_idx, dim in enumerate(dim_names):
                        dim_vals[dim][name] = features[i, d_idx]
                    break
    dataset.close()

    labels_maj = {k: emotion_map.get(v, v) for k, v in labels_maj.items()}
    labels_maj.update({k: "" for k in trn_df.index if k not in labels_maj})
    write_annotations(labels_maj, "label_maj")
    labels_plu = {k: emotion_map.get(v, v) for k, v in labels_plu.items()}
    labels_plu.update({k: "" for k in trn_df.index if k not in labels_plu})
    write_annotations(labels_plu, "label_plu")
    for dim in dim_names:
        dim_vals[dim].update({k: "" for k in trn_df.index if k not in dim_vals[dim]})
        write_annotations(dim_vals[dim], dim)
    labelled_paths = {resample_dir / f"{k}.flac" for k, v in labels_maj.items() if v}
    write_filelist(labelled_paths, "files_labels")

    fold_paths = defaultdict(list)
    vid_to_fold = {x: "train" for x in standard_train_fold}
    vid_to_fold.update({x: "valid" for x in standard_valid_fold})
    vid_to_fold.update({x: "test" for x in standard_test_fold})
    splits = {}
    for path in all_paths:
        video = path.stem.rsplit("_", maxsplit=1)[0]
        fold = vid_to_fold.get(video)
        splits[path.stem] = fold
        if fold is None:
            continue
        fold_paths[fold].append(path)
        if path in labelled_paths:
            fold_paths[f"{fold}_labels"].append(path)
    write_annotations(splits, "split")
    for key, pathlist in fold_paths.items():
        write_filelist(pathlist, f"files_{key}")


if __name__ == "__main__":
    main()
