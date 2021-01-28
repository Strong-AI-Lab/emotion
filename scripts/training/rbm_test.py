import argparse
from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf

from emotion_recognition.tensorflow.models import BBRBM, DecayType


def get_spectrograms(file, batch_size):
    dataset = netCDF4.Dataset(str(file))
    x = np.array(dataset.variables['features'])
    x = (x + 1.0) / 2.0
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    split = x.shape[0] // 4

    train = tf.data.Dataset.from_tensor_slices(x[split:, :])
    train = train.shuffle(1000)
    train = train.batch(batch_size).prefetch(10)

    test = tf.data.Dataset.from_tensor_slices(x[:split, :])
    test = test.batch(batch_size).prefetch(10)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='logs/rbm/', type=Path)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--init_learning_rate', default=0.005, type=float)
    parser.add_argument('--init_momentum', default=0.5, type=float)
    parser.add_argument('--final_learning_rate', default=0.001, type=float)
    parser.add_argument('--final_momentum', default=0.9, type=float)
    parser.add_argument('--learning_rate_decay', default=DecayType.STEP,
                        type=DecayType.__getitem__, choices=DecayType)
    parser.add_argument('--momentum_decay', default=DecayType.STEP,
                        type=DecayType.__getitem__, choices=DecayType)
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    train_data, test_data = get_spectrograms(
        'output/cafe/spectrograms-0.025-0.010-240-60.nc', args.batch_size)

    args.logdir.mkdir(parents=True, exist_ok=True)
    rbm = BBRBM(512, input_shape=(198, 240), logdir=args.logdir)
    rbm.train(
        train_data, test_data, n_epochs=200,
        init_learning_rate=args.init_learning_rate,
        init_momentum=args.init_momentum,
        final_learning_rate=args.final_learning_rate,
        final_momentum=args.final_momentum,
        learning_rate_decay=args.learning_rate_decay,
        momentum_decay=args.momentum_decay
    )


if __name__ == "__main__":
    main()
