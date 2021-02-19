"""Visualise 2D features for a single instance, or a collection of
feature vectors for multiple instances.
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from emorec.dataset import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--instance', type=str, default='2')
    args = parser.parse_args()

    dataset = Dataset(args.input)
    if args.instance.isdigit():
        instance = int(args.instance)
    else:
        idx = args.instance.find(':')
        if idx != -1:
            start = args.instance[:idx]
            end = args.instance[idx + 1:]
            if start.isdigit():
                start = int(start)
            if end.isdigit():
                end = int(end)
            instance = slice(start, end)
        else:
            instance = dataset.names.index(args.instance)
    arr = dataset.x[instance]
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, 0)

    names = dataset.names[instance]
    print(names)

    plt.figure()
    plt.imshow(arr, aspect='equal', origin='upper', interpolation='nearest')
    plt.xlabel('Features')
    plt.ylabel('Instance' if len(names) > 1 else 'Time')
    plt.show()


if __name__ == "__main__":
    main()
