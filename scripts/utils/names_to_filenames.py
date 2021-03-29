import argparse
from pathlib import Path

import pandas as pd
from emorec.dataset import get_audio_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", type=Path)
    parser.add_argument("--filenames", type=Path, required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.names.suffix == ".csv":
        df = pd.read_csv(args.names, header=0)
        names = list(df["Clip"])
    else:
        with open(args.names) as fid:
            names = list(map(str.strip, fid))

    paths = get_audio_paths(args.filenames)
    name_to_file = {p.stem: str(p) for p in paths}

    ordered = [name_to_file[n] for n in names if n in name_to_file]
    with open(args.output, "w") as fid:
        fid.write("\n".join(ordered) + "\n")


if __name__ == "__main__":
    main()
