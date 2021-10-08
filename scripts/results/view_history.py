from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.option("--individual", help="Plot individual folds for each metric.")
def main(input: Path, individual: bool):
    """Displays plot of training epochs for cross-validation rounds."""
    df = pd.read_csv(input, header=[0, 1], index_col=0)

    metrics = df.columns.get_level_values(1).unique()
    n_folds = len(df.columns.get_level_values(0).unique())
    metric_types = [x for x in metrics if not x.startswith("val_")]
    mean = df.mean(axis=1, level=1)
    std = df.std(axis=1, level=1)
    std_err = std / np.sqrt(n_folds)
    for metric in metric_types:
        cols = [metric, "val_" + metric]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("mean " + metric)
        ax.set_xlabel("epoch")
        for col in cols:
            x = mean.index
            y = mean[col]
            err = std_err[col]
            ax.plot(x, y, label="valid" if col.startswith("val_") else "train")
            ax.fill_between(x, y - 2 * err, y + 2 * err, alpha=0.2)
        ax.legend()

    if individual:
        metric_dfs = {}
        for key in df.columns.get_level_values(1).unique():
            metric_dfs[key] = df.xs(key, axis=1, level=1)
        for key, df in metric_dfs.items():
            df.plot(title=key)
    plt.show()


if __name__ == "__main__":
    main()
