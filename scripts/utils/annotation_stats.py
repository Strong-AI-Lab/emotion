from collections import Counter
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emorec.dataset import parse_annotations
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.option("--plot", is_flag=True, help="Plot histogram/bar chart.")
def main(input: Path, plot: bool):
    """Calculate statistics from annotations in INPUT."""
    _df = pd.read_csv(input, header=0, nrows=1)
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f"Distribution of {input.stem}")
        ax.set_xlabel(input.stem)
        ax.set_ylabel("Count")

    if _df.columns[1].lower() in ["speaker", "label"]:
        dict_ = parse_annotations(input, dtype=str)
        counts = Counter(dict_.values())
        print(f"{len(counts)} distinct values of {input.stem}:")
        for k, v in counts.most_common():
            print(f"{k}: {v}")
        if plot:
            ax.bar(counts.keys(), counts.values())
            ax.set_xticklabels(
                counts.keys(),
                rotation=30,
                ha="right",
                fontsize="small",
                rotation_mode="anchor",
            )
            fig.tight_layout()
    else:
        dict_ = parse_annotations(input)
        values = np.array(list(dict_.values()))
        print(f"Min: {np.min(values)}")
        print(f"Mean: {np.mean(values)}")
        print(f"Median: {np.median(values)}")
        print(f"Max: {np.max(values)}")
        print(f"Std: {np.std(values)}")
        if plot:
            ax.hist(values)
    plt.show()


if __name__ == "__main__":
    main()
