from pathlib import Path

import click
import pandas as pd
from emorec.utils import PathlibPath


@click.command()
@click.argument('input_dir', type=PathlibPath(exists=True, file_okay=False))
@click.argument('csv_file', type=Path, default='transcripts.csv')
def main(input_dir: Path, csv_file: Path):
    """Process EMO-DB transcripts.

    INPUT_DIR is the main EMO-DB directory containing the `silb/`
    directory.
    """
    utt = {}
    for p in input_dir.glob('silb/*.silb'):
        with open(p, encoding='latin_1') as fid:
            words = []
            for line in fid:
                line = line.strip()
                _, word = line.split()
                if word in ['.', '(']:
                    continue
                words.append(word.strip())
            utt[p.stem] = ' '.join(words)

    df = pd.DataFrame({'Name': utt.keys(), 'Transcript': utt.values()})
    df.sort_values('Name').to_csv(csv_file, index=False, header=True)
    print("Wrote CSV to {}".format(csv_file))


if __name__ == "__main__":
    main()
