import sys
from pathlib import Path
from typing import List

import click
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tqdm import tqdm

from ertk.dataset import read_features, write_features
from ertk.extraction import spectrogram
from ertk.utils import PathlibPath, batch_iterable, frame_array


def get_spec_generator(dataset):
    if dataset.features[0].shape[-1] == 1:
        # Raw audio input
        return (
            spectrogram(
                audio,
                sr=16000,
                pre_emphasis=0.97,
                window_size=0.025,
                window_shift=0.01,
                n_mels=64,
            )
            for audio in dataset.features
        )
    else:
        return (x for x in dataset.features)


def process_spec():
    pass


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
    input_layer = Input((96, 64))
    predictions, embeddings = yamnet(input_layer, Params())
    model = Model(inputs=input_layer, outputs=[predictions, embeddings])
    model.load_weights(model_path)
    model(tf.zeros((1, 96, 64)))

    print("Loading dataset")
    dataset = read_features(input)

    features: List[np.ndarray] = []
    batches = batch_iterable(get_spec_generator(dataset), batch_size)
    pbar = tqdm(total=len(dataset), desc="Processing frames")
    for batch in batches:
        batch = [x for x in batch if x is not None]

        frames = [frame_array(spec, 96, 48, pad=True, axis=0) for spec in batch]
        slices = np.cumsum([len(x) for x in frames])[:-1]
        frames = np.concatenate(frames)
        frames *= np.log(10.0) / 20  # Rescale dB power to ln(abs(X))

        _, embeddings = model(frames)
        features.extend(np.mean(x, 0) for x in np.split(embeddings, slices))

        pbar.update(len(batch))
    pbar.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"yamnet{i + 1}" for i in range(features[0].shape[-1])]
    write_features(
        output,
        names=dataset.names,
        features=features,
        corpus=dataset.corpus,
        feature_names=feature_names,
    )
    print("Wrote netCDF dataset to {}".format(output))


if __name__ == "__main__":
    main()
