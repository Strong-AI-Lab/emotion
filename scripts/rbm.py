from pathlib import Path

import netCDF4
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfd


def get_spectrograms(file):
    dataset = netCDF4.Dataset(str(file))
    x = np.array(dataset.variables['features'])
    x = (x + 1.0) / 2.0
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    split = x.shape[0] // 4

    train = tf.data.Dataset.from_tensor_slices(x[split:, :])
    train = train.shuffle(1000)
    train = train.batch(16).prefetch(1000)

    test = tf.data.Dataset.from_tensor_slices(x[:split, :])
    test = test.batch(16).prefetch(1000)
    return train, test


def get_mnist():
    mnist = tfd.load('mnist')
    train = mnist['train'].shuffle(10000)
    train = train.map(lambda x: tf.reshape(x['image'], (-1,)))
    train = train.map(lambda x: tf.cast(x, tf.float32) / 255.0)
    train = train.batch(32)
    test = mnist['test']
    test = test.map(lambda x: tf.reshape(x['image'], (-1,)))
    test = test.map(lambda x: tf.cast(x, tf.float32) / 255.0)
    test = test.batch(512)
    return train, test


def sample_bernoulli(p):
    return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))


class RBM:
    def __init__(self, n_hidden=128, input_shape=(100,), logdir='logs/rbm'):
        if any(x is None or x < 0 for x in input_shape):
            raise ValueError("input_shape must be fully specified.")

        self.input_shape = tf.TensorShape(input_shape)
        n_visible = tf.reduce_prod(self.input_shape)

        self.n_hidden = n_hidden

        self.h_bias = tf.Variable(tf.zeros((n_hidden,)))
        self.v_bias = tf.Variable(tf.zeros((n_visible,)))
        self.W = tf.Variable(tf.random.normal((n_visible, n_hidden), 0.0,
                                              0.01))

        self.h_bias_velocity = tf.Variable(tf.zeros_like(self.h_bias),
                                           trainable=False)
        self.v_bias_velocity = tf.Variable(tf.zeros_like(self.v_bias),
                                           trainable=False)
        self.W_velocity = tf.Variable(tf.zeros_like(self.W), trainable=False)

        self.learning_rate = tf.Variable(0.1, trainable=False)
        self.momentum = tf.Variable(0.5, trainable=False)
        self.l2_weight = tf.Variable(0.001, trainable=False)

        logdir = Path(logdir)
        timestamp = str(tf.timestamp().numpy())
        self.train_summary_writer = tf.summary.create_file_writer(
            str(logdir / timestamp / 'train'))
        self.valid_summary_writer = tf.summary.create_file_writer(
            str(logdir / timestamp / 'valid'))

    def log_free_energy(self, batch):
        pass

    def hidden_prob(self, batch):
        pass

    def calculate_deltas(self, batch):
        pass

    def train_batch(self, batch):
        pass

    def train(self, train_data, valid_data=None, init_learning_rate=0.1,
              init_momentum=0.9, n_epochs=100):
        self.learning_rate.assign(init_learning_rate)
        self.momentum.assign(init_momentum)
        for train_size, _ in enumerate(train_data):
            pass

        for epoch in range(1, n_epochs + 1):
            tf.print("Epoch: {}".format(epoch))

            mse_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            i = tf.constant(0, tf.int32)
            for batch in train_data:
                if i % (train_size // 10) == 0:
                    tf.print('.', end='')

                mse_batch = self.train_batch(batch)
                mse_train = mse_train.write(i, mse_batch)
                i += 1
            tf.print()

            mse_train = mse_train.stack()
            mse_train = tf.reduce_mean(mse_train)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('mse', mse_train, step=epoch)
                tf.print('mse: {:.3f}, '.format(mse_train))
                tf.summary.scalar('learining_rate', self.learning_rate,
                                  step=epoch)
                tf.summary.scalar('momentum', self.momentum, step=epoch)
                weights_l2 = tf.reduce_sum(self.W**2)
                tf.summary.scalar('weights_l2', weights_l2,
                                  step=epoch)

                # tf.summary.histogram('h_bias', self.h_bias, step=epoch)
                # tf.summary.histogram('v_bias', self.v_bias, step=epoch)
                # tf.summary.histogram('weights', self.W, step=epoch)

                # tf.summary.histogram('h_bias_velocity', self.h_bias_velocity,
                #                      step=epoch)
                # tf.summary.histogram('v_bias_velocity', self.v_bias_velocity,
                #                      step=epoch)
                # tf.summary.histogram('weights_velocity', self.W_velocity,
                #                      step=epoch)

                batch = next(iter(train_data))
                f_energy = self.log_free_energy(batch)
                tf.summary.scalar('log_free_energy', f_energy, step=epoch)

                new_shape = tf.concat(([self.n_hidden], self.input_shape, [1]),
                                      axis=0)
                W = tf.reshape(self.W, new_shape)
                W = (W + 2.0) / 4.0
                tf.summary.image('hidden_vis', W, step=epoch, max_outputs=128)

            if valid_data:
                mse_valid = tf.TensorArray(tf.float32, size=0,
                                           dynamic_size=True)
                i = tf.constant(0, tf.int32)
                for batch in valid_data:
                    p_v = self.reconstruct_batch(batch)
                    mse_batch = tf.reduce_mean((p_v - batch)**2)
                    mse_valid = mse_valid.write(i, mse_batch)
                    i += 1
                mse_valid = mse_valid.stack()
                mse_valid = tf.reduce_mean(mse_valid)

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar('mse', mse_valid, step=epoch)
                    tf.print('val_mse: {:.3f}, '.format(mse_valid), end='')

                    batch = next(iter(valid_data))
                    reconst = self.reconstruct_batch(batch)
                    new_shape = tf.concat(([-1], self.input_shape, [1]), 0)
                    reconst = tf.reshape(reconst, new_shape)
                    image = tf.reshape(batch, new_shape)
                    image = tf.concat([image, reconst], 2)
                    tf.summary.image('reconstruction', image, step=epoch,
                                     max_outputs=5)

                    p_h = self.hidden_prob(batch)
                    p_h = tf.reshape(p_h, (1, -1, self.n_hidden, 1))
                    p_h = tf.transpose(p_h, [0, 2, 1, 3])
                    tf.summary.image('hidden_prob', p_h, step=epoch)

                    f_energy = self.log_free_energy(batch)
                    tf.summary.scalar('log_free_energy', f_energy,
                                      step=epoch)

    def reconstruct(self, test_data):
        return test_data.map(self.reconstruct_batch)


class BBRBM(RBM):
    def log_free_energy(self, batch):
        with tf.device('/device:GPU:0'):
            x = self.h_bias + tf.matmul(batch, self.W)
            x_64 = tf.cast(x, tf.float64)
            log_64 = tf.math.log1p(tf.exp(x_64), -1)
            log_32 = tf.cast(log_64, tf.float32)
            log_f_energy = (- tf.reduce_sum(batch * self.v_bias, -1)
                            - tf.reduce_sum(log_32, -1))
            log_f_energy = tf.reduce_mean(log_f_energy)
        return log_f_energy

    def hidden_prob(self, batch):
        return tf.sigmoid(self.h_bias + tf.matmul(batch, self.W))

    def visible_prob(self, batch, h):
        return tf.sigmoid(self.v_bias + tf.matmul(h, self.W, transpose_b=True))

    @tf.function
    def train_batch(self, batch):
        with tf.device('/device:GPU:0'):
            p_h = self.hidden_prob(batch)
            h = sample_bernoulli(p_h)
            p_v = self.visible_prob(batch, h)
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

            self.W_velocity.assign(self.momentum * self.W_velocity
                                   + self.learning_rate * W_delta)
            self.h_bias_velocity.assign(self.momentum * self.h_bias_velocity
                                        + self.learning_rate * h_bias_delta)
            self.v_bias_velocity.assign(self.momentum * self.v_bias_velocity
                                        + self.learning_rate * v_bias_delta)

            self.W.assign_add(self.W_velocity)
            self.h_bias.assign_add(self.h_bias_velocity)
            self.v_bias.assign_add(self.v_bias_velocity)

            mse_batch = tf.reduce_mean((p_v - batch)**2)
        return mse_batch

    @tf.function
    def reconstruct_batch(self, batch):
        with tf.device('/device:GPU:0'):
            p_h = self.hidden_prob(batch)
            h = sample_bernoulli(p_h)
            p_v = self.visible_prob(batch, h)
        return p_v


class GBRBM(RBM):
    def log_free_energy(self, batch):
        x = self.h_bias + tf.matmul(batch, self.W)
        x_64 = tf.cast(x, tf.float64)
        log_64 = tf.math.log1p(tf.exp(x_64), -1)
        log_32 = tf.cast(log_64, tf.float32)
        log_f_energy = (- tf.reduce_sum(batch * self.v_bias, -1)
                        - tf.reduce_sum(log_32, -1))
        log_f_energy = tf.reduce_mean(log_f_energy)
        return log_f_energy

    def hidden_prob(self, batch):
        return tf.sigmoid(self.h_bias + tf.matmul(batch, self.W))

    def visible_prob(self, batch, h):
        return self.v_bias + tf.matmul(h, self.W, transpose_b=True)

    @tf.function
    def train_batch(self, batch):
        with tf.device('/device:GPU:0'):
            p_h = self.hidden_prob(batch)
            h = sample_bernoulli(p_h)
            p_v = self.visible_prob(batch, h)
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

            self.W_velocity.assign(self.momentum * self.W_velocity
                                   + self.learning_rate * W_delta)
            self.h_bias_velocity.assign(self.momentum * self.h_bias_velocity
                                        + self.learning_rate * h_bias_delta)
            self.v_bias_velocity.assign(self.momentum * self.v_bias_velocity
                                        + self.learning_rate * v_bias_delta)

            self.W.assign_add(self.W_velocity)
            self.h_bias.assign_add(self.h_bias_velocity)
            self.v_bias.assign_add(self.v_bias_velocity)

            mse_batch = tf.reduce_mean((p_v - batch)**2)
        return mse_batch

    @tf.function
    def reconstruct_batch(self, batch):
        with tf.device('/device:GPU:0'):
            p_h = self.hidden_prob(batch)
            h = sample_bernoulli(p_h)
            p_v = self.visible_prob(batch, h)
        return p_v


def main():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.debugging.set_log_device_placement(False)

    train_data, test_data = get_spectrograms(
        'audeep/spectrograms/cafe-0.05-0.025-240-60.nc')

    rbm = BBRBM(128, input_shape=(198, 240), logdir='logs/rbm')
    rbm.train(train_data, test_data, n_epochs=100, init_learning_rate=0.5,
              init_momentum=0.5)


if __name__ == "__main__":
    main()
