from itertools import chain
from pathlib import Path
from typing import List, Tuple

import click

from ertk.dataset import read_features_iterable, write_features


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1
)
@click.argument("output", type=Path)
@click.option(
    "--prefix_corpus", is_flag=True, help="Prefix corpus names to instance names."
)
@click.option("--corpus", default="combined", help="Output corpus name.")
def main(input: Tuple[Path], output: Path, prefix_corpus: bool, corpus: str):
    """Combines multiple INPUT features and writes to OUTPUT."""

    if len(input) == 0:
        raise ValueError("No input files specified.")

    feature_names: List[str] = []
    features = []
    names = []
    total_length = 0
    for filename in input:
        data = read_features_iterable(filename)
        total_length += len(data)
        if len(feature_names) == 0:
            feature_names = list(data.feature_names)
        elif len(data.feature_names) != len(feature_names):
            raise ValueError("Feature size of all datasets must match.")
        features.append(iter(data))
        if prefix_corpus:
            names += [f"{data.corpus}_{x}" for x in data.names]
        else:
            names += data.names
    if len(set(names)) != total_length:
        raise ValueError(
            "Some names are not unique, use --prefix_corpus to generate unique names."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_features(
        output,
        chain.from_iterable(features),
        corpus=corpus,
        names=names,
        feature_names=feature_names,
    )
    print(f"Wrote combined features to {output}")


if __name__ == "__main__":
    main()
