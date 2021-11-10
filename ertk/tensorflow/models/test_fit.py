from pathlib import Path

import click
import numpy as np

from . import get_tf_model
from ..utils import tf_dataset_mem, tf_dataset_gen, init_gpu_memory_growth
from ...utils import get_arg_mapping


@click.command()
@click.argument("model_name")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
@click.option(
    "--summary_only", is_flag=True, help="Print summary but don't train on dummy data."
)
@click.option(
    "--memory/--generated",
    default=False,
    help="Have all data in memory or generate batches.",
)
@click.option("--model_args", type=click.Path(exists=True, path_type=Path))
@click.option("--train", type=int, default=8000, help="Number of training instances.")
@click.option("--valid", type=int, default=800, help="Number of validation instances.")
@click.option("--features", type=int, default=1024, help="Dimensionality of input.")
@click.option("--steps", type=int, default=512, help="Length of sequence.")
def test_fit(
    model_name: str,
    batch_size: int,
    summary_only: bool,
    memory: bool,
    model_args: Path,
    train: int,
    valid: int,
    features: int,
    steps: int,
):
    """Creates a TensorFlow model and fits it with random data. This is
    useful for seeing how much memory will be used with different batch
    sizes and how long training takes.
    """

    init_gpu_memory_growth()

    args = get_arg_mapping(model_args) if model_args else {}
    model = get_tf_model(model_name, n_features=features, n_classes=4, **args)
    model.summary()

    if summary_only:
        return

    shape = (features,) if len(model.input_shape) == 2 else (steps, features)

    rng = np.random.default_rng()
    train_x = rng.random((train,) + shape, dtype=np.float32)
    train_y = rng.integers(4, size=train)
    valid_x = rng.random((valid,) + shape, dtype=np.float32)
    valid_y = rng.integers(4, size=valid)

    if memory:
        train_data = tf_dataset_mem(
            train_x, train_y, batch_size=batch_size, shuffle=True
        )
        valid_data = tf_dataset_mem(
            valid_x, valid_y, batch_size=batch_size, shuffle=False
        )
    else:
        train_data = tf_dataset_gen(
            train_x, train_y, batch_size=batch_size, shuffle=True
        )
        valid_data = tf_dataset_gen(
            valid_x, valid_y, batch_size=batch_size, shuffle=False
        )

    model.compile(loss="sparse_categorical_crossentropy")
    model.fit(train_data, epochs=2, validation_data=valid_data, verbose=True)


if __name__ == "__main__":
    test_fit()
