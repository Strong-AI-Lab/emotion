from pathlib import Path

import click

from ertk.dataset import read_features_iterable, write_features


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=Path)
@click.option("--corpus", type=str, help="Corpus attribute to set, if required.")
def main(input: Path, output: Path, corpus: str):
    """Convert INPUT dataset format to OUTPUT format. Note that no label
    information is written to OUTPUT.
    """

    if input.suffix == output.suffix:
        raise ValueError("Input format must be different to output.")

    print(f"Reading {input}")
    data = read_features_iterable(input)
    if corpus:
        data.corpus = corpus
    output.parent.mkdir(parents=True, exist_ok=True)
    write_features(
        output,
        iter(data),
        names=data.names,
        corpus=data.corpus,
        feature_names=data.feature_names,
    )
    print(f"Wrote dataset to {output}")


if __name__ == "__main__":
    main()
