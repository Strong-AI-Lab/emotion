from pathlib import Path
from typing import Union

import click
import matplotlib.pyplot as plt

from emorec.dataset import read_features
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("instance", type=str, default="2")
def main(input: Path, instance: str):
    """Displays plot of INSTANCE in INPUT. INSTANCE can either be a
    numeric index, a range of indices using numpy slice notation or a
    named instance.
    """

    data = read_features(input)
    if instance.isdigit():
        idx: Union[int, slice] = int(instance)
    else:
        _i = instance.find(":")
        if _i != -1:
            start = int(instance[:_i])
            end = int(instance[_i + 1 :])
            idx = slice(start, end)
        else:
            idx = data.names.index(instance)
    arr = data.features[idx]
    names = data.names[idx]
    print(names)

    plt.figure()
    plt.imshow(arr, aspect="equal", origin="upper", interpolation="nearest")
    plt.xlabel("Features")
    plt.ylabel("Instance" if len(names) > 1 else "Time")
    plt.show()


if __name__ == "__main__":
    main()
