"""A TensorFlow 2 implementation of the auDeep representation learning
framework. Original implementation is available at
https://github.com/auDeep/auDeep.

References:
[1] S. Amiriparian, M. Freitag, N. Cummins, and B. Schuller, 'Sequence
to sequence autoencoders for unsupervised representation learning from
audio', 2017.
[2] M. Freitag, S. Amiriparian, S. Pugachevskiy, N. Cummins, and B.
Schuller, 'auDeep: Unsupervised learning of representations from audio
with deep recurrent neural networks', The Journal of Machine Learning
Research, vol. 18, no. 1, pp. 6340–6344, 2017.
"""

import argparse
import time
from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from emorec.tensorflow.models import audeep_trae


def _make_functions(model, optimizer, strategy, use_function=True):
    def train_step(data):
        with tf.GradientTape() as tape:
            reconstruction, _ = model(data, training=True)
            targets = data[:, ::-1, :]
            loss = tf.sqrt(
                tf.reduce_mean(tf.math.squared_difference(targets, reconstruction))
            )

        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        clipped_gvs = [tf.clip_by_value(x, -2, 2) for x in gradients]
        # clipped_gvs, _ = tf.clip_by_global_norm(gradients, 2)
        optimizer.apply_gradients(zip(clipped_gvs, trainable_vars))

        return loss

    def test_step(data):
        reconstruction, _ = model(data, training=False)
        targets = data[:, ::-1, :]
        loss = tf.sqrt(
            tf.reduce_mean(tf.math.squared_difference(targets, reconstruction))
        )

        return loss

    def dist_train_step(batch):
        per_replica_losses = strategy.run(train_step, args=(batch,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    def dist_test_step(batch):
        per_replica_losses = strategy.run(test_step, args=(batch,))
        return strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
        )

    if use_function:
        dist_train_step = tf.function(dist_train_step)
        dist_test_step = tf.function(dist_test_step)

    return dist_train_step, dist_test_step


def train(args):
    args.logs.parent.mkdir(parents=True, exist_ok=True)
    train_log_dir = str(args.logs / "train")
    valid_log_dir = str(args.logs / "valid")
    train_writer = tf.summary.create_file_writer(train_log_dir)
    valid_writer = tf.summary.create_file_writer(valid_log_dir)

    if args.profile:
        tf.profiler.experimental.start("")
        tf.profiler.experimental.stop(save=False)

    strategy = tf.distribute.get_strategy()
    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy()

    # Get data
    dataset = netCDF4.Dataset(str(args.dataset))
    x = np.array(dataset.variables["features"])
    dataset.close()
    n_spectrograms, max_time, features = x.shape
    np.random.default_rng().shuffle(x)
    n_valid = int(n_spectrograms * args.valid_fraction)
    valid_data = (
        tf.data.Dataset.from_tensor_slices(x[:n_valid])
        .batch(args.batch_size)
        .prefetch(10)
    )
    train_data = (
        tf.data.Dataset.from_tensor_slices(x[n_valid:])
        .shuffle(n_spectrograms)
        .batch(args.batch_size)
        .prefetch(100)
    )
    summary_data = valid_data.take(1)
    # Reduce memory footprint
    del x

    train_data = strategy.experimental_distribute_dataset(train_data)
    valid_data = strategy.experimental_distribute_dataset(valid_data)

    n_train_batches = len(train_data)
    n_valid_batches = len(valid_data)

    step = tf.Variable(1)
    with strategy.scope():
        model = audeep_trae(
            (max_time, features),
            units=args.units,
            layers=args.layers,
            bidirectional_encoder=args.bidirectional_encoder,
            bidirectional_decoder=args.bidirectional_decoder,
        )
        # Using epsilon=1e-5 seems to offer better stability.
        optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate, epsilon=1e-5)
        model.compile(optimizer=optimizer)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
    print()
    print("Model structure:")
    for layer in model.layers:
        name = layer.name
        if name.startswith("tf_op_layer_"):
            name = name[12:]
        print(layer.name, layer.output_shape)
    print()

    # Write the graph only
    tf.keras.callbacks.TensorBoard(
        args.logs / "graph", write_graph=True, profile_batch=0
    ).set_model(model)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        str(args.logs),
        max_to_keep=args.keep_checkpoints,
        checkpoint_interval=args.save_interval,
        step_counter=step,
    )
    if args.cont:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restoring from checkpoint " f"{checkpoint_manager.latest_checkpoint}")

    train_step, test_step = _make_functions(model, optimizer, strategy)

    start_epoch = step.value().numpy()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        step.assign(epoch)
        epoch_time = time.perf_counter()
        train_loss = 0.0
        n_batch = 0
        batch_time = time.perf_counter()
        for batch in tqdm(
            train_data, total=n_train_batches, desc=f"Epoch {epoch:03d}", unit="batch"
        ):
            if epoch == 2 and n_batch == 0 and args.profile:
                tf.profiler.experimental.start(args.logdir)
            loss = train_step(batch)
            train_loss += loss
            n_batch += 1
            if epoch == 2 and n_batch == 0 and args.profile:
                tf.profiler.experimental.stop()
        train_loss /= n_train_batches
        batch_time = time.perf_counter() - batch_time
        batch_time /= n_train_batches

        valid_loss = 0.0
        for batch in valid_data:
            loss = test_step(batch)
            valid_loss += loss
        valid_loss /= n_valid_batches

        epoch_time = time.perf_counter() - epoch_time
        print(
            f"Epoch {epoch:03d}: train loss: {train_loss:.4f}, valid loss: "
            f"{valid_loss:.4f} in {epoch_time:.2f}s ({batch_time:.2f}s per "
            "batch)."
        )

        with valid_writer.as_default():
            tf.summary.scalar("rmse", valid_loss, step=epoch)

            batch = next(iter(summary_data))
            reconstruction, representation = model(batch, training=False)
            reconstruction = reconstruction[:, ::-1, :]
            images = tf.concat([batch, reconstruction], 2)
            images = tf.expand_dims(images, -1)
            images = (images + 1) / 2
            tf.summary.image("combined", images, step=epoch, max_outputs=20)
            tf.summary.histogram("representation", representation, step=epoch)

        with train_writer.as_default():
            tf.summary.scalar("rmse", train_loss, step=epoch)

        if epoch % args.save_interval == 0:
            checkpoint_manager.save(checkpoint_number=epoch)
            print(f"Saved checkpoint to " f"{checkpoint_manager.latest_checkpoint}")

    save_path = str(args.logs / "model")
    model.save(save_path, include_optimizer=False)
    print(f"Saved model to {save_path}")


def generate(args):
    dataset = netCDF4.Dataset(args.dataset)
    data = tf.data.Dataset.from_tensor_slices(dataset.variables["features"]).batch(
        args.batch_size
    )
    filenames = np.array(dataset.variables["name"])
    labels = np.array(dataset.variables["label"])
    corpus = dataset.corpus
    dataset.close()

    model = load_model(args.model)
    # Optimise model call function
    model.call = tf.function(model.call)

    print(f"Read dataset from {args.dataset}")

    representations = []
    for batch in tqdm(data):
        _, representation = model(batch, training=False)
        representations.append(representation.numpy())
    representations = np.concatenate(representations)

    dataset = netCDF4.Dataset(str(args.output), "w")
    dataset.createDimension("instance", len(filenames))
    dataset.createDimension("generated", representations.shape[-1])

    filename = dataset.createVariable("name", str, ("instance",))
    filename[:] = filenames

    label_nominal = dataset.createVariable("label", str, ("instance",))
    label_nominal[:] = labels

    features = dataset.createVariable("features", np.float32, ("instance", "generated"))
    features[:, :] = representations

    dataset.setncattr_string("feature_dims", '["generated"]')
    dataset.setncattr_string("corpus", corpus)
    dataset.close()

    print(f"Wrote netCDF4 file to {args.output}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="command", dest="command", required=True)

    train_args = subparsers.add_parser("train")
    train_args.add_argument(
        "--dataset", type=Path, required=True, help="File containing spectrogram data."
    )
    train_args.add_argument(
        "--logs", type=Path, required=True, help="Directory to store TensorBoard logs."
    )

    train_args.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train for."
    )
    train_args.add_argument(
        "--valid_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use as validation data.",
    )
    train_args.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    train_args.add_argument(
        "--batch_size", type=int, default=64, help="Global batch size."
    )
    train_args.add_argument(
        "--multi_gpu", action="store_true", help="Use all GPUs for training."
    )
    train_args.add_argument("--profile", action="store_true", help="Profile a batch.")
    train_args.add_argument(
        "--continue",
        action="store_true",
        dest="cont",
        help="Continue from latest checkpoint.",
    )
    train_args.add_argument(
        "--keep_checkpoints", type=int, default=5, help="Number of checkpoints to keep."
    )
    train_args.add_argument(
        "--save_interval", type=int, default=10, help="Save every N epochs."
    )

    train_args.add_argument(
        "--units", type=int, default=256, help="Dimensionality of RNN cells."
    )
    train_args.add_argument(
        "--layers", type=int, default=2, help="Number of stacked RNN layers."
    )
    train_args.add_argument(
        "--bidirectional_encoder",
        action="store_true",
        help="Use a bidirectional encoder.",
    )
    train_args.add_argument(
        "--bidirectional_decoder",
        action="store_true",
        help="Use a bidirectional decoder.",
    )

    gen_args = subparsers.add_parser("generate")
    gen_args.add_argument(
        "--dataset", type=Path, required=True, help="File containing spectrogram data."
    )
    gen_args.add_argument(
        "--model", type=Path, required=True, help="Directory of Keras saved_model."
    )
    gen_args.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    gen_args.add_argument(
        "--output", type=Path, required=True, help="Path to output representations."
    )

    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)


if __name__ == "__main__":
    main()
