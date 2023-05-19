from enum import Enum
from math import pi
from pathlib import Path
from typing import Optional, Sequence, Tuple

import tensorflow as tf

from ertk.utils import PathOrStr

__all__ = ["BBRBM", "GBRBM"]


def _sample_bernoulli(p: tf.Tensor):
    return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))


def _sample_std_normal(p: tf.Tensor):
    return tf.random.normal()


class DecayType(Enum):
    STEP = "STEP"
    COSINE = "COSINE"
    EXP = "EXP"

    def __str__(self):
        return self.name


class RBM:
    """Restricted Boltzmann Machine

    This implementation is designed around the work of Hinton et. al.
    [1], [2].

    References
    ----------
    [1] G. E. Hinton and R. R. Salakhutdinov, 'Reducing the
    Dimensionality of Data with Neural Networks', Science, vol. 313, no.
    5786, pp. 504-507, Jul. 2006, doi: 10.1126/science.1127647.

    [2] G. E. Hinton, 'A Practical Guide to Training Restricted
    Boltzmann Machines', in Neural Networks: Tricks of the Trade: Second
    Edition, G. Montavon, G. B. Orr, and K.-R. Müller, Eds. Berlin,
    Heidelberg: Springer Berlin Heidelberg, 2012, pp. 599-619.
    """

    def __init__(
        self,
        n_hidden: int,
        input_shape: Tuple[int, ...],
        logdir: PathOrStr = "logs/rbm",
        output_histograms: bool = False,
    ):
        """
        Parameters
        ----------
        n_hidden: int, default = 128
            Number of hidden units
        input_shape: tuple
            Shape of input tensors, either 1D or 2D
        logdir: str or Path, optional
            Path to store TensorBoard logs. Default is 'logs/rbm'.
        output_histograms: bool, optional
            Whether to output histogram information. Can slow down training
            somewhat. Default is False.
        """
        if any(x is None or x < 0 for x in input_shape):
            raise ValueError("input_shape must be fully specified.")

        self.input_shape = tf.TensorShape(input_shape)
        n_visible = tf.reduce_prod(self.input_shape)

        self.n_hidden = n_hidden

        self.h_bias = tf.Variable(tf.zeros((n_hidden,)))
        self.v_bias = tf.Variable(tf.zeros((n_visible,)))
        self.W = tf.Variable(tf.random.normal((n_visible, n_hidden), 0.0, 0.01))

        self.h_bias_velocity = tf.Variable(tf.zeros_like(self.h_bias), trainable=False)
        self.v_bias_velocity = tf.Variable(tf.zeros_like(self.v_bias), trainable=False)
        self.W_velocity = tf.Variable(tf.zeros_like(self.W), trainable=False)

        self.learning_rate = tf.Variable(0.01, trainable=False)
        self.momentum = tf.Variable(0.5, trainable=False)
        self.l2_weight = tf.Variable(0.001, trainable=False)

        self.output_histograms = output_histograms

        self.logging = False
        if logdir:
            self.logging = True
            logdir = Path(logdir)
            self.train_summary_writer = tf.summary.create_file_writer(
                str(logdir / "train")
            )
            self.valid_summary_writer = tf.summary.create_file_writer(
                str(logdir / "valid")
            )

    def free_energy(self, batch: tf.Tensor):
        raise NotImplementedError("free_energy() not implemented.")

    def hidden_prob(self, batch: tf.Tensor):
        raise NotImplementedError("hidden_prob() not implemented.")

    def visible_prob(self, h: tf.Tensor):
        raise NotImplementedError("visible_prob() not implemented.")

    def sample_h(self, p: tf.Tensor):
        raise NotImplementedError("sample_h() not implemented.")

    def sample_v(self, p: tf.Tensor):
        raise NotImplementedError("sample_v() not implemented.")

    def calculate_deltas(self, batch: tf.Tensor):
        raise NotImplementedError("calculate_deltas() not implemented.")

    def forward(self, batch: tf.Tensor):
        """Samples hidden units from input batch.

        Parameters
        ----------
        batch: tensor of shape (batch, n_visible)
            Batch of tensors to process.

        Returns
        -------
        Sample hidden units.
        """
        p_h = self.hidden_prob(batch)
        h = self.sample_h(p_h)
        return h

    def backward(self, h: tf.Tensor):
        p_v = self.visible_prob(h)
        return p_v

    @tf.function
    def train_batch(self, batch: tf.Tensor):
        """Runs one step of Gibbs sampling and gradient ascent with the given
        batch.

        Parameters
        ----------
        batch: tensor of shape (batch, n_visible)
            Batch of tensors to train on.

        Returns
        -------
        Mean square error (MSE) for this batch.
        """
        p_h = self.hidden_prob(batch)
        h = self.sample_h(p_h)
        p_v = self.visible_prob(h)
        p_hd = tf.sigmoid(self.h_bias + tf.matmul(p_v, self.W))

        # v*h.T - v'*h'.T
        pos = tf.matmul(tf.expand_dims(batch, -1), tf.expand_dims(p_h, -2))
        neg = tf.matmul(tf.expand_dims(p_v, -1), tf.expand_dims(p_hd, -2))
        outers = pos - neg
        W_delta = tf.reduce_mean(outers, 0) - self.l2_weight * 2.0 * self.W
        # h - h'
        h_bias_delta = tf.reduce_mean(p_h - p_hd, 0)
        # v - v'
        v_bias_delta = tf.reduce_mean(batch - p_v, 0)

        self.W_velocity.assign(
            self.momentum * self.W_velocity + self.learning_rate * W_delta
        )
        self.h_bias_velocity.assign(
            self.momentum * self.h_bias_velocity + self.learning_rate * h_bias_delta
        )
        self.v_bias_velocity.assign(
            self.momentum * self.v_bias_velocity + self.learning_rate * v_bias_delta
        )

        self.W.assign_add(self.W_velocity)
        self.h_bias.assign_add(self.h_bias_velocity)
        self.v_bias.assign_add(self.v_bias_velocity)

        mse_batch = tf.reduce_mean((p_v - batch) ** 2)
        return mse_batch

    def train(
        self,
        train_data: tf.data.Dataset,
        valid_data: Optional[tf.data.Dataset] = None,
        init_learning_rate: float = 0.01,
        final_learning_rate: float = 0.001,
        learning_rate_decay: Optional[DecayType] = None,
        learning_rate_decay_param: float = 0.5,
        init_momentum: float = 0.5,
        final_momentum: float = 0.9,
        momentum_decay: Optional[DecayType] = None,
        momentum_decay_param: float = 0.5,
        n_epochs: int = 100,
    ):
        """Train this RBM.

        Parameters
        ----------
        train_data: tf.data.Dataset
            Training data to use.
        valid_data: tf.data.Dataset, optional
            Validation data to use for performance measuring.
        init_learning_rate: float
            The initial learning rate to set. Default is 0.01.
        final_learning_rate: float
            The final learning rate to set. Default is 0.001
        learning_rate_decay: DecayType, optional
            Type of learning rate decay. Default is None, meaning no decay.
        learning_rate_decay_param: float
            Parameter for learning rate decay. Use depends on type. For step
            decay, this is the fraction of epochs after which to change
            learning rate.
        init_momentum: float
            The initial momentum to set. Default is 0.5.
        final_momentum: float
            The final momentum to set. Default is 0.9.
        momentum_decay: DecayType, optional
            Type of momentum decay. Default is None, meaning no decay.
        n_epochs: int
            The number of epochs to train for. A single epoch is one pass over
            the training data. Default is 100.
        """
        self.learning_rate.assign(init_learning_rate)
        self.momentum.assign(init_momentum)
        train_size = sum(1 for _ in train_data)

        if learning_rate_decay == DecayType.COSINE:
            lr_coeffA = 0.5 * (init_learning_rate - final_learning_rate)
            lr_coeffB = 0.5 * (init_learning_rate + final_learning_rate)
        elif learning_rate_decay == DecayType.EXP:
            lr_coeffA = init_learning_rate
            lr_coeffB = tf.math.log(init_learning_rate) - tf.math.log(
                final_learning_rate
            )
        if momentum_decay == DecayType.COSINE:
            mom_coeffA = 0.5 * (init_momentum - final_momentum)
            mom_coeffB = 0.5 * (init_momentum + final_momentum)
        elif momentum_decay == DecayType.EXP:
            mom_coeffA = init_momentum
            mom_coeffB = tf.math.log(init_momentum) - tf.math.log(final_momentum)

        for epoch in range(1, n_epochs + 1):
            tf.print(f"Epoch: {epoch}")
            time_frac = float(epoch) / float(n_epochs)

            if (
                learning_rate_decay == DecayType.STEP
                and time_frac > learning_rate_decay_param
            ):
                self.learning_rate.assign(final_learning_rate)
            elif learning_rate_decay == DecayType.COSINE:
                self.learning_rate.assign(
                    lr_coeffA * tf.cos(pi * time_frac) + lr_coeffB
                )
            elif learning_rate_decay == DecayType.EXP:
                self.learning_rate.assign(lr_coeffA * tf.exp(-lr_coeffB * time_frac))

            if momentum_decay == DecayType.STEP and time_frac > momentum_decay_param:
                self.momentum.assign(final_momentum)
            elif momentum_decay == DecayType.COSINE:
                self.momentum.assign(mom_coeffA * tf.cos(pi * time_frac) + mom_coeffB)
            elif momentum_decay == DecayType.EXP:
                self.momentum.assign(mom_coeffA * tf.exp(-mom_coeffB * time_frac))

            mse_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            i = tf.constant(0, tf.int32)
            for batch in train_data:
                if i % (train_size // 10) == 0:
                    tf.print(".", end="")

                mse_batch = self.train_batch(batch)
                mse_train = mse_train.write(i, mse_batch)
                i += 1

            mse_train = mse_train.stack()
            mse_train = tf.reduce_mean(mse_train)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("mse", mse_train, step=epoch)
                tf.print(f"mse: {mse_train:.3f}, ", end="")

                tf.summary.scalar("learining_rate", self.learning_rate, step=epoch)
                tf.summary.scalar("momentum", self.momentum, step=epoch)
                weights_l2 = tf.reduce_sum(self.W ** 2)
                tf.summary.scalar("weights_l2", weights_l2, step=epoch)

                if self.output_histograms:
                    tf.summary.histogram("h_bias", self.h_bias, step=epoch)
                    tf.summary.histogram("v_bias", self.v_bias, step=epoch)
                    tf.summary.histogram("weights", self.W, step=epoch)

                    tf.summary.histogram(
                        "h_bias_velocity", self.h_bias_velocity, step=epoch
                    )
                    tf.summary.histogram(
                        "v_bias_velocity", self.v_bias_velocity, step=epoch
                    )
                    tf.summary.histogram(
                        "weights_velocity", self.W_velocity, step=epoch
                    )

                batch = next(iter(train_data))
                f_energy = self.free_energy(batch)
                tf.summary.scalar("free_energy", f_energy, step=epoch)

                if 2 <= len(self.input_shape) <= 3:
                    if len(self.input_shape) == 3:
                        new_shape = tf.concat(
                            ([self.n_hidden], self.input_shape), axis=0
                        )
                    elif len(self.input_shape) == 2:
                        new_shape = tf.concat(
                            ([self.n_hidden], self.input_shape, [1]), axis=0
                        )
                    W = tf.transpose(self.W)
                    W = tf.reshape(W, new_shape)
                    W = (W + 2.0) / 4.0
                    tf.summary.image("hidden_vis", W, step=epoch, max_outputs=128)

                    if len(self.input_shape) == 3:
                        new_shape = tf.concat(([-1], self.input_shape), axis=0)
                    elif len(self.input_shape) == 2:
                        new_shape = tf.concat(([-1], self.input_shape, [1]), axis=0)
                    reconst = self.reconstruct_batch(batch)
                    reconst = tf.reshape(reconst, new_shape)
                    image = tf.reshape(batch, new_shape)
                    combined = tf.concat([image, reconst], 2)
                    tf.summary.image(
                        "reconstruction", combined, step=epoch, max_outputs=10
                    )

            if valid_data:
                mse_valid = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                i = tf.constant(0, tf.int32)
                for batch in valid_data:
                    p_v = self.reconstruct_batch(batch)
                    mse_batch = tf.reduce_mean((p_v - batch) ** 2)
                    mse_valid = mse_valid.write(i, mse_batch)
                    i += 1
                mse_valid = mse_valid.stack()
                mse_valid = tf.reduce_mean(mse_valid)

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("mse", mse_valid, step=epoch)
                    tf.print(f"val_mse: {mse_valid:.3f}")

                    batch = next(iter(valid_data))

                    if 2 <= len(self.input_shape) <= 3:
                        if len(self.input_shape) == 3:
                            new_shape = tf.concat(([-1], self.input_shape), axis=0)
                        elif len(self.input_shape) == 2:
                            new_shape = tf.concat(([-1], self.input_shape, [1]), axis=0)
                        reconst = self.reconstruct_batch(batch)
                        reconst = tf.reshape(reconst, new_shape)
                        image = tf.reshape(batch, new_shape)
                        combined = tf.concat([image, reconst], 2)
                        tf.summary.image(
                            "reconstruction", combined, step=epoch, max_outputs=10
                        )

                    p_h = self.hidden_prob(batch)
                    p_h = tf.reshape(p_h, (1, -1, self.n_hidden, 1))
                    p_h = tf.transpose(p_h, [0, 2, 1, 3])
                    tf.summary.image("hidden_prob", p_h, step=epoch)

                    f_energy = self.free_energy(batch)
                    tf.summary.scalar("free_energy", f_energy, step=epoch)

    @tf.function
    def reconstruct_batch(self, batch):
        """Reconstruct a single batch of input.

        Parameters
        ----------
        batch: tensor of shape (batch, n_visible)
            The batch of tensor to reconstruct.

        Returns
        -------
        Reconstructed batch.
        """
        p_h = self.hidden_prob(batch)
        h = self.sample_h(p_h)
        p_v = self.visible_prob(h)
        return p_v

    def reconstruct(self, data: tf.data.Dataset):
        """Reconstruct all input data.

        Parameters
        ----------
        data: tf.data.Dataset
            Data to reconstruct.

        Returns
        -------
        The reconstructed data as a mapped tf.data.Dataset.
        """
        return data.map(self.reconstruct_batch)

    def representations(self, data: tf.data.Dataset):
        return data.map(self.hidden_prob)


class BBRBM(RBM):
    """RBM with Bernoulli visible and hidden units."""

    def free_energy(self, batch: tf.Tensor):
        x = self.h_bias + tf.matmul(batch, self.W)
        f_energy = -tf.reduce_sum(batch * self.v_bias, -1) - tf.reduce_sum(
            tf.nn.softplus(x), -1
        )
        f_energy = tf.reduce_mean(f_energy)
        return f_energy

    def hidden_prob(self, batch: tf.Tensor):
        return tf.sigmoid(self.h_bias + tf.matmul(batch, self.W))

    def visible_prob(self, h: tf.Tensor):
        return tf.sigmoid(self.v_bias + tf.matmul(h, self.W, transpose_b=True))

    def sample_h(self, p: tf.Tensor):
        return _sample_bernoulli(p)


class GBRBM(RBM):
    """RBM with Gaussian visible and Bernoulli hidden units."""

    def free_energy(self, batch: tf.Tensor):
        x = self.h_bias + tf.matmul(batch, self.W)
        f_energy = 0.5 * tf.reduce_sum((batch - self.v_bias) ** 2, -1) - tf.reduce_sum(
            tf.nn.softplus(x), -1
        )
        f_energy = tf.reduce_mean(f_energy)
        return f_energy

    def hidden_prob(self, batch: tf.Tensor):
        return tf.sigmoid(self.h_bias + tf.matmul(batch, self.W))

    def visible_prob(self, h: tf.Tensor):
        return self.v_bias + tf.matmul(h, self.W, transpose_b=True)

    def sample_h(self, p: tf.Tensor):
        return _sample_bernoulli(p)


class DBN:
    """Deep Belief Network, also known as a Deep Boltzmann Machine."""

    def __init__(
        self,
        input_shape: Tuple[int],
        n_layers: int = 3,
        layer_nodes: Sequence[int] = [512, 256, 128],
        logdir: PathOrStr = "logs/dbn",
        **kwargs,
    ):
        """Initialise this DBN.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input tensors to the first layer
        n_layers: int
            Number of hidden layers in this DBN.
        layer_nodes: tuple of int
            Number of hidden nodes in each of the layers. We must have that
            len(layer_nodes) == n_layers
        logdir: str or Path
            Directory to save log files.
        kwargs: dict
            Additional arguments to pass to the RBM constructor.
        """
        if n_layers != len(layer_nodes):
            raise ValueError("layer_nodes must have exactly n_layers elements")

        self.input_shape = input_shape

        logdir = Path(logdir)
        timestamp = tf.timestamp().numpy()
        logdir = logdir / str(timestamp)
        self.layers = [
            BBRBM(layer_nodes[0], self.input_shape, logdir=logdir / "rbm0", **kwargs)
        ]
        for i in range(1, n_layers):
            self.layers.append(
                BBRBM(
                    layer_nodes[i],
                    logdir=logdir / f"rbm{i}",
                    input_shape=(layer_nodes[i - 1],),
                )
            )

        self.train_summary_writer = tf.summary.create_file_writer(
            str(logdir / "dbn" / "train")
        )
        self.valid_summary_writer = tf.summary.create_file_writer(
            str(logdir / "dbn" / "valid")
        )

    def train(
        self,
        train_data: tf.data.Dataset,
        valid_data: Optional[tf.data.Dataset] = None,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
    ):
        """Train each layer in this DBN in sequence, using the reconstructed
        output from previous layers as input.

        Parameters
        ----------
        train_data: tf.data.Dataset
            Training data to use.
        valid_data: tf.data.Dataset, optional
            Validation data to use for mesuring training performance.
        n_epochs: int
            The number of epochs to train each layer for. Default is 100.
        learning_rate: float
            The learning rate to use for all layers. Default is 0.01.
        momentum: float
            The momentum to use for all layers. Default is 0.5.
        """
        print("Training layer 0")
        self.layers[0].train(
            train_data,
            valid_data=valid_data,
            init_momentum=momentum,
            init_learning_rate=learning_rate,
            n_epochs=n_epochs,
        )
        new_train_data = train_data
        new_valid_data = valid_data
        for i in range(1, len(self.layers)):
            print(f"Training layer {i}")
            new_train_data = new_train_data.map(self.layers[i - 1].forward)
            if new_valid_data:
                new_valid_data = new_valid_data.map(self.layers[i - 1].forward)
            self.layers[i].train(
                new_train_data,
                valid_data=new_valid_data,
                init_momentum=momentum,
                init_learning_rate=learning_rate,
                n_epochs=n_epochs,
            )

        if valid_data:
            with self.valid_summary_writer.as_default():
                batch = next(iter(valid_data))

                if len(self.input_shape) == 3:
                    new_shape = tf.concat(([-1], self.input_shape), axis=0)
                elif len(self.input_shape) == 2:
                    new_shape = tf.concat(([-1], self.input_shape, [1]), axis=0)
                tf.print("Reconstructing")
                reconst = self.reconstruct_batch(batch)
                reconst = tf.reshape(reconst, new_shape)
                image = tf.reshape(batch, new_shape)
                combined = tf.concat([image, reconst], 2)
                tf.summary.image(
                    "reconstruction", combined, step=n_epochs, max_outputs=10
                )

    @tf.function
    def reconstruct_batch(self, batch: tf.Tensor):
        """Reconstruct a single batch of input.

        Parameters
        ----------
        batch: tensor of shape (batch, n_visible)
            The batch of tensor to reconstruct.

        Returns
        -------
        Reconstructed batch.
        """
        for i in range(len(self.layers)):
            batch = self.layers[i].forward(batch)
        for i in range(len(self.layers) - 1, -1, -1):
            batch = self.layers[i].backward(batch)
        return batch

    def reconstruct(self, data: tf.data.Dataset):
        """Reconstruct all input data.

        Parameters
        ----------
        data: tf.data.Dataset
            Data to reconstruct.

        Returns
        -------
        The reconstructed data as a mapped tf.data.Dataset.
        """
        return data.map(self.reconstruct_batch)
