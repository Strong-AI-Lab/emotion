from pathlib import Path
from typing import List, Tuple

import click

from ertk.dataset import read_features, write_features


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1
)
@click.argument("output", type=Path)
@click.option(
    "--prefix_corpus", is_flag=True, help="Prefix corpus names to instance names."
)
def main(input: Tuple[Path], output: Path, prefix_corpus: bool):
    """Combines multiple INPUT features and writes to OUTPUT."""

    if len(input) == 0:
        raise ValueError("No input files specified.")

    feature_names: List[str] = []
    features = []
    names = []
    for filename in input:
        data = read_features(filename)
        if len(feature_names) == 0:
            feature_names = list(data.feature_names)
        elif len(data.feature_names) != len(feature_names):
            raise ValueError("Feature size of all datasets must match.")
        features += list(data.features)
        if prefix_corpus:
            names += [f"{data.corpus}_{x}" for x in data.names]
        else:
            names += data.names
    if len(set(names)) != len(features):
        raise ValueError(
            "Some names are not unique, use --prefix_corpus to generate unique names."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_features(
        output,
        features,
        corpus="combined",
        names=names,
        feature_names=feature_names,
    )
    print(f"Wrote combined features to {output}")


if __name__ == "__main__":
    main()
