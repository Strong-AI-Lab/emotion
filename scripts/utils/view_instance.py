import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

import click
import numpy as np
from emorec.dataset import Dataset
from emorec.utils import PathlibPath


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.argument('instance', type=str, default="2")
def main(input: Path, instance: str):
    """Displays plot of INSTANCE in INPUT. INSTANCE can either be a
    numeric index, a range of indices using numpy slice notation or a
    named instance.
    """

    dataset = Dataset(input)
    if instance.isdigit():
        idx: Union[int, slice] = int(instance)
    else:
        _i = instance.find(':')
        if _i != -1:
            start = int(instance[:_i])
            end = int(instance[_i + 1:])
            idx = slice(start, end)
        else:
            idx = dataset.names.index(instance)
    arr = dataset.x[idx]
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, 0)

    names = dataset.names[idx]
    print(names)

    plt.figure()
    plt.imshow(arr, aspect='equal', origin='upper', interpolation='nearest')
    plt.xlabel('Features')
    plt.ylabel('Instance' if len(names) > 1 else 'Time')
    plt.show()


if __name__ == "__main__":
    main()
