# NOTE: This file is not part of the TensorFlow VGGish models repository

import sys
from pathlib import Path

import click
import numpy as np
import tensorflow.compat.v1 as tf

from ertk.dataset import Dataset, write_features

# isort: off
from vggish_postprocess import Postprocessor
from vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint
# isort: on


@click.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
@click.option("--batch_size", type=int, default=256)
def main(input: Path, output: Path, batch_size: int):
    """Extracts the VGGish embeddings from a set of spectrograms in
    INPUT and writes a new dataset to OUTPUT.
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Loading dataset")
    dataset = Dataset(input)
    frames_list = [tf.signal.frame(x, 96, 96, pad_end=True, axis=0) for x in dataset.x]
    slices = tf.constant([len(x) for x in frames_list])
    frames = tf.concat(frames_list, 0)
    frames *= tf.math.log(10.0) / 20  # Rescale dB power to ln(abs(X))
    frames = frames.numpy()

    print("Loading VGGish model")
    with tf.Graph().as_default(), tf.Session() as sess:
        define_vggish_slim()
        ckpt = str(Path(sys.argv[0]).parent / "vggish_model.ckpt")
        load_vggish_slim_checkpoint(sess, ckpt)

        features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
        embedding_tensor = sess.graph.get_tensor_by_name("vggish/embedding:0")

        print("Processing frames")
        outputs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            [embeddings] = sess.run(
                [embedding_tensor], feed_dict={features_tensor: batch}
            )
            outputs.append(embeddings)
        outputs = np.concatenate(outputs, 0)

    print("Postprocessing")
    pca_params = str(Path(sys.argv[0]).parent / "vggish_pca_params.npz")
    processor = Postprocessor(pca_params)
    embeddings = processor.postprocess(outputs)
    slices = np.cumsum(slices)[:-1]
    embeddings = [np.mean(x, 0) for x in np.split(embeddings, slices)]
    embeddings = np.stack(embeddings, 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [f"vggish{i + 1}" for i in range(embeddings.shape[-1])]
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
