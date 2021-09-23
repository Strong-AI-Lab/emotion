from pathlib import Path

import click
import numpy as np
import tensorflow as tf

from ertk.dataset import Dataset, write_features
from ertk.tensorflow.models.rbm import BBRBM, DecayType
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.option("--logdir", type=str, default="logs/tf/rbm")
@click.option("--batch_size", type=int, default=16)
@click.option("--valid_fraction", type=float, default=0.1)
@click.option("--epochs", type=int, default=200)
@click.option("--init_learning_rate", type=float, default=0.005)
@click.option("--init_momentum", type=float, default=0.5)
@click.option("--final_learning_rate", type=float, default=0.001)
@click.option("--final_momentum", type=float, default=0.9)
@click.option(
    "--learning_rate_decay",
    type=click.Choice(DecayType.__members__),
    default=DecayType.STEP.name,
)
@click.option(
    "--momentum_decay",
    type=click.Choice(DecayType.__members__),
    default=DecayType.STEP.name,
)
@click.option("--output", type=Path)
def main(
    input: Path,
    logdir: str,
    batch_size: int,
    valid_fraction: float,
    epochs: int,
    init_learning_rate: float,
    init_momentum: float,
    final_learning_rate: float,
    final_momentum: float,
    learning_rate_decay: DecayType,
    momentum_decay: DecayType,
    output: Path,
):
    """Trains RBM on spectrograms in INPUT."""

    learning_rate_decay = DecayType(learning_rate_decay)
    momentum_decay = DecayType(momentum_decay)

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = Dataset(input)
    x = dataset.x
    input_shape = tuple(x[0].shape)
    xmin = np.min(x, axis=(1, 2), keepdims=True)
    xmax = np.max(x, axis=(1, 2), keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    spectrograms = np.reshape(x, (dataset.n_instances, -1))

    split = int(spectrograms.shape[0] * valid_fraction)
    train_data = tf.data.Dataset.from_tensor_slices(spectrograms[split:, :])
    train_data = train_data.shuffle(spectrograms.shape[0])
    train_data = train_data.batch(batch_size).prefetch(10)
    valid_data = tf.data.Dataset.from_tensor_slices(spectrograms[:split, :])
    valid_data = valid_data.batch(batch_size).prefetch(10)

    Path(logdir).mkdir(parents=True, exist_ok=True)
    rbm = BBRBM(512, input_shape=input_shape, logdir=logdir, output_histograms=True)
    rbm.train(
        train_data,
        valid_data,
        n_epochs=epochs,
        init_learning_rate=init_learning_rate,
        init_momentum=init_momentum,
        final_learning_rate=final_learning_rate,
        final_momentum=final_momentum,
        learning_rate_decay=learning_rate_decay,
        momentum_decay=momentum_decay,
    )
    if output:
        representations = rbm.representations(
            tf.data.Dataset.from_tensor_slices(spectrograms).batch(8)
        )
        representations = np.concatenate(list(representations.as_numpy_iterator()))
        feature_names = [f"bbrbm{i + 1}" for i in range(representations.shape[-1])]
        write_features(
            output,
            names=dataset.names,
            features=representations,
            corpus=dataset.corpus,
            feature_names=feature_names,
        )


if __name__ == "__main__":
    main()
