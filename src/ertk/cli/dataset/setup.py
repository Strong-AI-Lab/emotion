from pathlib import Path

import click

from ertk.dataset.predefined import list_predefined_datasets, process_dataset


@click.command()
@click.argument("dataset", type=click.Choice(list_predefined_datasets()))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--resample/--no-resample", default=True)
def main(
    dataset: str, input_dir: Path, output_dir: Path, resample: bool = True
) -> None:
    """Set up a pre-defined dataset."""

    process_dataset(dataset, input_dir, output_dir, resample)


if __name__ == "__main__":
    main()
