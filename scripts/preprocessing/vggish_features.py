import sys
from pathlib import Path
from typing import List

import click
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from ertk.dataset import read_features, write_features
from ertk.preprocessing import spectrogram
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


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
@click.option("--batch_size", type=int, default=8)
@click.option(
    "--vggish",
    "vggish_dir",
    type=PathlibPath(exists=True, file_okay=False),
    default=Path("third_party", "models", "audioset", "vggish"),
    show_default=True,
)
def main(input: Path, output: Path, batch_size: int, vggish_dir: Path):
    """Extracts the VGGish embeddings from a set of spectrograms in
    INPUT and writes a new dataset to OUTPUT.
    """
    sys.path.insert(0, str(vggish_dir))
    try:
        from vggish_postprocess import Postprocessor
        from vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot find VGGish code, make sure the correct path is given"
        ) from e

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Loading dataset")
    dataset = read_features(input)

    print("Loading VGGish model")
    pca_params = str(vggish_dir / "vggish_pca_params.npz")
    processor = Postprocessor(pca_params)

    features: List[np.ndarray] = []
    batches = batch_iterable(get_spec_generator(dataset), batch_size)
    pbar = tqdm(total=len(dataset), desc="Processing frames")
    with tf.Graph().as_default(), tf.Session() as sess:
        define_vggish_slim()
        ckpt = str(vggish_dir / "vggish_model.ckpt")
        load_vggish_slim_checkpoint(sess, ckpt)

        features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
        embedding_tensor = sess.graph.get_tensor_by_name("vggish/embedding:0")

        for batch in batches:
            batch = [x for x in batch if x is not None]

            frames = [frame_array(spec, 96, 96, pad=True, axis=0) for spec in batch]
            slices = np.cumsum([len(x) for x in frames])[:-1]
            frames = np.concatenate(frames)
            frames *= np.log(10.0) / 20  # Rescale dB power to ln(abs(X))
            [embeddings] = sess.run(
                [embedding_tensor], feed_dict={features_tensor: frames}
            )
            embeddings = processor.postprocess(embeddings)
            features.extend(np.mean(x, 0) for x in np.split(embeddings, slices))

            pbar.update(len(batch))
        pbar.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"vggish{i + 1}" for i in range(embeddings[0].shape[-1])]
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
