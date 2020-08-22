import argparse
import time
from IPython.core.debugger import set_trace
from pathlib import Path

import netCDF4
import numpy as np
import tensorboard
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Workaround for having both PyTorch and TensorFlow installed.
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class TimeRecurrentAutoencoder(nn.Module):
    def __init__(self, n_features: int,
                 n_units: int = 256,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional_encoder: bool = False,
                 bidirectional_decoder: bool = False):
        super().__init__()

        encoder_state_size = n_units * n_layers
        decoder_state_size = n_units * n_layers
        if bidirectional_encoder:
            encoder_state_size *= 2
        if bidirectional_decoder:
            decoder_state_size *= 2
        decoder_output_size = n_units * n_layers  # concat(fwd, back)

        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder

        self.encoder = nn.GRU(
            n_features, n_units, num_layers=n_layers, dropout=dropout,
            bidirectional=bidirectional_encoder, batch_first=True
        )
        self.representation = nn.Sequential(
            nn.Linear(encoder_state_size, decoder_state_size),
            nn.Tanh()
        )
        self.decoder = nn.GRU(
            n_features, n_units, num_layers=n_layers, dropout=dropout,
            bidirectional=bidirectional_decoder, batch_first=True
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(decoder_output_size, n_features), nn.Tanh())

    def forward(self, x: torch.Tensor):
        targets = x.flip(1)

        num_directions = 2 if self.bidirectional_encoder else 1
        encoder_h = torch.zeros(self.n_layers * num_directions, x.size(0),
                                self.n_units, device=x.device)
        _, encoder_h = self.encoder(
            F.dropout(x, self.dropout, self.training), encoder_h)
        encoder_h = torch.cat(encoder_h.unbind(), -1)

        representation = self.representation(encoder_h)

        num_directions = 2 if self.bidirectional_decoder else 1
        decoder_h = torch.stack(representation.chunk(
            self.n_layers * num_directions, -1))

        decoder_input = F.pad(targets[:, :-1, :], [0, 0, 1, 0])
        decoder_seq, _ = self.decoder(
            F.dropout(decoder_input, self.dropout, self.training), decoder_h)
        decoder_seq = F.dropout(decoder_seq, self.dropout, self.training)

        reconstruction = self.reconstruction(decoder_seq)
        reconstruction = reconstruction.flip(1)
        return reconstruction, representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)

    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bidirectional_encoder', action='store_true')
    parser.add_argument('--bidirectional_decoder', action='store_true')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--valid_fraction', type=float, default=0.2)
    args = parser.parse_args()

    DEVICE = torch.device('cuda:0')

    dataset = netCDF4.Dataset(args.input)
    x = np.array(dataset.variables['features'])
    dataset.close()
    np.random.default_rng().shuffle(x)

    x = torch.tensor(x).to(DEVICE)

    n_valid = int(args.valid_fraction * x.size(0))
    train_data = TensorDataset(x[n_valid:])
    valid_data = TensorDataset(x[:n_valid])

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = TimeRecurrentAutoencoder(
        x.size(2), n_units=args.units, n_layers=args.layers,
        dropout=args.dropout, bidirectional_encoder=args.bidirectional_encoder,
        bidirectional_decoder=args.bidirectional_decoder
    )
    model = model.cuda(DEVICE)

    args.logs.mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(args.logs / 'train')
    valid_writer = SummaryWriter(args.logs / 'valid')

    with torch.no_grad():
        batch = next(iter(train_dl))
        train_writer.add_graph(model, input_to_model=batch)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        model.train()
        losses = []
        for x, in train_dl:
            optimizer.zero_grad()
            reconstruction, representation = model(x)

            loss = loss_fn(x, reconstruction)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        train_time = time.perf_counter() - start_time
        train_loss = np.mean(losses)

        model.eval()
        with torch.no_grad():
            losses = []
            it = iter(valid_dl)

            # Get loss and write summaries for first batch
            x, = next(it)
            reconstruction, representation = model(x)
            loss = loss_fn(x, reconstruction)
            losses.append(loss.item())

            images = torch.cat([x, reconstruction], 2).unsqueeze(-1)
            images = (images + 1) / 2
            images = images[:20]
            valid_writer.add_images('reconstruction', images,
                                    global_step=epoch, dataformats='NHWC')
            valid_writer.add_histogram('representation', representation,
                                       global_step=epoch)

            # Just get the loss for the rest, no summaries
            for x, in it:
                reconstruction, representation = model(x)
                loss = loss_fn(x, reconstruction)
                losses.append(loss.item())
        valid_loss = np.mean(losses)

        train_writer.add_scalar('loss', train_loss, global_step=epoch)
        valid_writer.add_scalar('loss', valid_loss, global_step=epoch)

        end_time = time.perf_counter() - start_time

        print(
            "Epoch {:03d} in {:.2f}s ({:.2f}s per batch), train loss: {:.4f}, "
            "valid loss: {:.4f}".format(
                epoch, end_time, train_time / len(train_dl), train_loss,
                valid_loss
            )
        )

    # Generate

    # Open dataset again
    dataset = netCDF4.Dataset(args.input)
    x = np.array(dataset.variables['features'])
    filenames = np.array(dataset.variables['filename'])
    labels = np.array(dataset.variables['label_nominal'])
    corpus = dataset.corpus
    dataset.close()

    representations = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(x), args.batch_size):
            batch = torch.tensor(x[i:i + args.batch_size], device=DEVICE)
            _, representation = model(batch)
            representations.append(representation.cpu().numpy())
    representations = np.concatenate(representations)

    train_writer.add_embedding(representations, labels)

    dataset = netCDF4.Dataset(args.output, 'w')
    dataset.createDimension('instance', len(filenames))
    dataset.createDimension('generated', representations.shape[-1])

    filename = dataset.createVariable('filename', str, ('instance',))
    filename[:] = filenames

    label_nominal = dataset.createVariable('label_nominal', str, ('instance',))
    label_nominal[:] = labels

    features = dataset.createVariable('features', np.float32,
                                      ('instance', 'generated'))
    features[:, :] = representations

    dataset.setncattr_string('feature_dims', '["generated"]')
    dataset.setncattr_string('corpus', corpus)
    dataset.close()

    print("Wrote netCDF4 file to {}.".format(args.output))


if __name__ == "__main__":
    main()
