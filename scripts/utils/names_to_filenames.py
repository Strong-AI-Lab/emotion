import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=Path, required=True)
    parser.add_argument('--filenames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.names.suffix == '.csv':
        df = pd.read_csv(args.names, header=None)
        names = list(df[0])
    else:
        with open(args.names) as fid:
            names = list(map(str.strip, fid))

    with open(args.filenames) as fid:
        name_to_file = {Path(x.strip()).stem: x.strip() for x in fid}

    ordered = [name_to_file[n] for n in names if n in name_to_file]
    with open(args.output, 'w') as fid:
        fid.write('\n'.join(ordered))


if __name__ == "__main__":
    main()
