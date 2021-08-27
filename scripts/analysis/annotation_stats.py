from pathlib import Path
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import pandas as pd

from ertk.dataset import get_audio_paths
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--plot", is_flag=True, help="Plot histogram/bar chart.")
@click.option(
    "--files",
    type=PathlibPath(exists=True, dir_okay=False),
    help="File with names to include for statistics.",
)
def main(input: Tuple[Path], plot: bool, files: Path):
    """Calculate statistics from annotations in INPUT(s). If multiple
    INPUTs are given, additional statistics for the other INPUTs are
    shown for each level of INPUT.
    """

    dfs: List[pd.DataFrame] = []
    for file in input:
        df = pd.read_csv(file, header=0, converters={0: str}).set_index("name")
        if files:
            names = {Path(x).stem for x in get_audio_paths(files)}
            df = df[df.index.isin(names)]
        dfs.append(df)
        col = df[df.columns[0]]
        print(f"== Statistics for {col.name} ==")
        print(col.describe().to_string())
        print()
        if col.dtype == object:
            counts = col.value_counts()
            print("Value counts:")
            print(counts.to_string())
            print()

        if plot:
            fig, ax = plt.subplots()
            ax.set_title(f"Distribution of {file.stem}")
            if col.dtype == object:
                col.value_counts().plot(kind="bar", ax=ax)
            else:
                col.plot(kind="hist", ylabel=None, ax=ax)
            fig.tight_layout()

    if len(input) > 1:
        print("== Pairwise tables ==")
        joined = dfs[0]
        refcol = joined.columns[0]
        for df in dfs[1:]:
            joined = joined.join(df)
        for col in joined.columns[1:]:
            print(f"For column '{col}':")
            if plot:
                fig, ax = plt.subplots()
                ax.set_title(f"Distribution of {col} per {refcol}")

            if joined[refcol].dtype == object:
                if joined[col].dtype == object:
                    table = joined.groupby([refcol, col]).size().unstack(fill_value=0)
                    print(table.to_string())
                    if plot:
                        table.plot(kind="bar", stacked=True, ax=ax)
                    if len(joined.columns) > 2:
                        uniq = (
                            joined.groupby([refcol, col])
                            .nunique()
                            .unstack(fill_value=0)
                        )
                        print("Unique")
                        print(uniq.to_string())
                else:
                    print(joined.groupby([refcol])[col].describe())
                    if plot:
                        table = joined.groupby([refcol])[col].hist(
                            bins=5, alpha=0.3, legend=True, ax=ax
                        )
            else:
                joined.plot(kind="scatter", x=refcol, y=col, ax=ax)
            print()
            if plot:
                fig.tight_layout()

    if plot:
        plt.show()


if __name__ == "__main__":
    main()
