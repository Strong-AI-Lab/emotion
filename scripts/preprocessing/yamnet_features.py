import sys
from pathlib import Path

import click
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tqdm import trange

from ertk.dataset import read_features, write_features
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
@click.option("--batch_size", type=int, default=256)
@click.option(
    "--yamnet",
    "yamnet_dir",
    type=PathlibPath(exists=True, file_okay=False),
    default=Path("third_party", "models", "audioset", "yamnet"),
    show_default=True,
)
def main(input: Path, output: Path, batch_size: int, yamnet_dir: Path):
    """Extracts the YAMNet embeddings from a set of spectrograms in
    INPUT and writes a new dataset to OUTPUT.
    """
    sys.path.insert(0, str(yamnet_dir))
    try:
        from params import Params
        from yamnet import yamnet
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot find YAMNet code, make sure the correct path is given"
        ) from e

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model_path = yamnet_dir / "yamnet.h5"
    print(f"Loading yamnet model {model_path}")
    features = Input((96, 64))
    predictions, embeddings = yamnet(features, Params())
    model = Model(inputs=features, outputs=[predictions, embeddings])
    model.load_weights(model_path)
    model(tf.zeros((1, 96, 64)))

    print(f"Loading dataset {input}")
    dataset = read_features(input)
    frames_list = [
        tf.signal.frame(x, 96, 48, pad_end=True, axis=0) for x in dataset.features
    ]
    slices = tf.constant([len(x) for x in frames_list])
    frames = tf.concat(frames_list, 0)
    frames *= tf.math.log(10.0) / 20  # Rescale dB power to ln(abs(X))

    print(f"Processing {len(frames)} frames")
    outputs = []
    for i in trange(0, len(frames), batch_size):
        _, embeddings = model(frames[i : i + batch_size])
        outputs.append(embeddings)
    embeddings = tf.concat(outputs, 0)
    embeddings = [tf.reduce_mean(x, 0).numpy() for x in tf.split(embeddings, slices)]

    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"yamnet{i + 1}" for i in range(embeddings[0].shape[-1])]
    write_features(
        output,
        names=dataset.names,
        features=embeddings,
        corpus=dataset.corpus,
        feature_names=feature_names,
    )
    print("Wrote netCDF dataset to {}".format(output))


if __name__ == "__main__":
    main()
