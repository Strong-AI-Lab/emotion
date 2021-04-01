# NOTE: This file is not part of the TensorFlow YAMNet models repository

import sys
from pathlib import Path

import click
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from emorec.dataset import Dataset, write_netcdf_dataset

# isort: off
from params import Params
from yamnet import yamnet
# isort: on


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
@click.option("--batch_size", type=int, default=256)
def main(input: Path, output: Path, batch_size: int):
    """Extracts the YAMNet embeddings from a set of spectrograms in
    INPUT and writes a new dataset to OUTPUT.
    """

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model_path = Path(sys.argv[0]).parent / "yamnet.h5"
    print(f"Loading yamnet model {model_path}")
    features = Input((96, 64))
    predictions, embeddings = yamnet(features, Params())
    model = Model(inputs=features, outputs=[predictions, embeddings])
    model.load_weights(model_path)
    model(tf.zeros((1, 96, 64)))

    print(f"Loading dataset {input}")
    dataset = Dataset(input)
    frames_list = [tf.signal.frame(x, 96, 48, pad_end=True, axis=0) for x in dataset.x]
    slices = tf.constant([len(x) for x in frames_list])
    frames = tf.concat(frames_list, 0)
    frames *= tf.math.log(10.0) / 20  # Rescale dB power to ln(abs(X))

    print(f"Processing {len(frames)} frames")
    outputs = []
    for i in range(0, len(frames), batch_size):
        _, embeddings = model(frames[i : i + batch_size])
        outputs.append(embeddings)
    embeddings = tf.concat(outputs, 0)
    embeddings = [tf.reduce_mean(x, 0) for x in tf.split(embeddings, slices)]
    embeddings = tf.stack(embeddings, 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"yamnet{i + 1}" for i in range(embeddings.shape[-1])]
    write_netcdf_dataset(
        output,
        dataset.names,
        embeddings,
        corpus=dataset.corpus,
        feature_names=feature_names,
    )
    print("Wrote netCDF dataset to {}".format(output))


if __name__ == "__main__":
    main()
