import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ertk.dataset import read_features, write_features
from ertk.utils import PathOrStr

DEVICE = torch.device("cuda")


class TimeRecurrentAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_units: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
    ):
        super().__init__()

        encoder_state_size = n_units * n_layers
        decoder_state_size = n_units * n_layers
        if bidirectional_encoder:
            encoder_state_size *= 2
        if bidirectional_decoder:
            decoder_state_size *= 2
        decoder_output_size = n_units * n_layers  # concat(fwd, back)
        self.embedding_size = decoder_state_size

        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder

        self.encoder = nn.GRU(
            n_features,
            n_units,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
            batch_first=True,
        )
        self.representation = nn.Sequential(
            nn.Linear(encoder_state_size, decoder_state_size), nn.Tanh()
        )
        self.decoder = nn.GRU(
            n_features,
            n_units,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional_decoder,
            batch_first=True,
        )
        self.reconstruction = nn.Sequential(
            nn.Linear(decoder_output_size, n_features), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x (B, T, F)
        targets = x.flip(1)

        num_directions = 2 if self.bidirectional_encoder else 1
        encoder_h = torch.zeros(
            self.n_layers * num_directions, x.size(0), self.n_units, device=x.device
        )
        _, encoder_h = self.encoder(
            F.dropout(x, self.dropout, self.training), encoder_h
        )
        encoder_h = torch.cat(encoder_h.unbind(), -1)

        representation = self.representation(encoder_h)

        num_directions = 2 if self.bidirectional_decoder else 1
        decoder_h = torch.stack(
            representation.chunk(self.n_layers * num_directions, -1)
        )

        decoder_input = F.pad(targets[:, :-1, :], [0, 0, 1, 0])
        decoder_seq, _ = self.decoder(
            F.dropout(decoder_input, self.dropout, self.training), decoder_h
        )
        decoder_seq = F.dropout(decoder_seq, self.dropout, self.training)

        reconstruction = self.reconstruction(decoder_seq)
        reconstruction = reconstruction.flip(1)
        return reconstruction, representation


def rmse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error loss function."""
    return torch.sqrt(torch.mean((x - y) ** 2))


def get_latest_save(directory: Union[str, Path]) -> Path:
    """Gets the path to the latest model checkpoint, assuming it was saved with
    the format 'model-XXXX.pt'

    Parameters
    ----------
    directory: Path or str
        The path to a directory containing the model save files.

    Returns:
        The path to the latest model save file (i.e. the one with the highest
        epoch number).
    """
    directory = Path(directory)
    saves = list(directory.glob("model-*.pt"))
    if len(saves) == 0:
        raise FileNotFoundError(f"No save files found in directory {directory}")
    return max(saves, key=lambda p: int(p.stem.split("-")[1]))


def save_model(
    directory: Union[Path, str],
    epoch: int,
    model: TimeRecurrentAutoencoder,
    model_args: Dict[str, Any],
    optimiser: Adam,
    optimiser_args: Dict[str, Any],
):
    """Saves the given model and associated training parameters."""
    save_path = Path(directory) / f"model-{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_args": model_args,
            "optimiser_state_dict": optimiser.state_dict(),
            "optimiser_args": optimiser_args,
        },
        save_path,
    )
    print(f"Saved model to {save_path}.")


def get_data(path: PathOrStr, shuffle: bool = False):
    """Get data and metadata from netCDF dataset. Optionally shuffle x."""
    dataset = read_features(path)
    x = dataset.features
    names = np.array(dataset.names)

    if shuffle:
        perm = np.random.default_rng().permutation(len(x))
        x = x[perm]
        names = names[perm]

    x = scale(x, -1, 1)
    return x, names, dataset.corpus


def scale(x, low: float = -1, high: float = 1):
    xmin = np.min(x, axis=(1, 2), keepdims=True)
    xmax = np.max(x, axis=(1, 2), keepdims=True)
    return (high - low) * (x - xmin) / (xmax - xmin) + low


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--logs", type=Path, required=True)
@click.option("--layers", type=int, default=2)
@click.option("--units", type=int, default=256)
@click.option("--dropout", type=float, default=0.2)
@click.option("--bidirectional_encoder", is_flag=True)
@click.option("--bidirectional_decoder", is_flag=True)
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=50)
@click.option("--learning_rate", type=float, default=0.001)
@click.option("--valid_fraction", type=float, default=0.1)
@click.option("--continue", "cont", is_flag=True)
@click.option("--save_every", type=int, default=20)
def train(
    input: Path,
    logs: Path,
    layers: int,
    units: int,
    dropout: float,
    bidirectional_encoder: bool,
    bidirectional_decoder: bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    valid_fraction: float,
    cont: bool,
    save_every: int,
):
    print("Reading input from {}".format(input))
    x = get_data(input, shuffle=True)[0]
    x = torch.tensor(x)

    n_valid = int(valid_fraction * x.size(0))
    train_data = TensorDataset(x[n_valid:])
    valid_data = TensorDataset(x[:n_valid])
    train_dl = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_dl = DataLoader(valid_data, batch_size=batch_size, pin_memory=True)

    if cont:
        save_path = get_latest_save(logs)
        load_dict = torch.load(save_path)
        model_args = load_dict["model_args"]
        optimiser_args = load_dict["optimiser_args"]
        initial_epoch = 1 + load_dict["epoch"]
    else:
        model_args = dict(
            n_features=x.size(2),
            n_units=units,
            n_layers=layers,
            dropout=dropout,
            bidirectional_encoder=bidirectional_encoder,
            bidirectional_decoder=bidirectional_decoder,
        )
        optimiser_args = dict(lr=learning_rate, eps=1e-5)
        initial_epoch = 1
    final_epoch = initial_epoch + epochs - 1

    model = TimeRecurrentAutoencoder(**model_args)
    model = model.cuda(DEVICE)
    optimiser = Adam(model.parameters(), **optimiser_args)

    if cont:
        print(f"Restoring model from {save_path}")
        model.load_state_dict(load_dict["model_state_dict"])
        optimiser.load_state_dict(load_dict["optimiser_state_dict"])

    logs.mkdir(parents=True, exist_ok=True)
    train_writer = SummaryWriter(logs / "train")
    valid_writer = SummaryWriter(logs / "valid")
    with torch.no_grad():
        batch = next(iter(train_dl))[0].to(DEVICE)
        train_writer.add_graph(model, input_to_model=batch)

    loss_fn = rmse_loss
    for epoch in range(initial_epoch, final_epoch + 1):
        start_time = time.perf_counter()
        model.train()
        train_losses = []
        for (x,) in tqdm(
            train_dl, desc=f"Epoch {epoch:03d}", unit="batch", leave=False
        ):
            x = x.to(DEVICE)
            optimiser.zero_grad()
            reconstruction, representation = model(x)

            loss = loss_fn(x, reconstruction)
            train_losses.append(loss.item())

            loss.backward()
            # Clip gradients to [-2, 2]
            nn.utils.clip_grad_value_(model.parameters(), 2)
            optimiser.step()
        train_time = time.perf_counter() - start_time
        train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            valid_losses = []
            it = iter(valid_dl)

            # Get loss and write summaries for first batch
            x = next(it)[0].to(DEVICE)
            reconstruction, representation = model(x)
            loss = loss_fn(x, reconstruction)
            valid_losses.append(loss.item())

            if epoch % 10 == 0:
                images = torch.cat([x, reconstruction], 2).unsqueeze(-1)
                images = (images + 1) / 2
                images = images[:20]
                valid_writer.add_images(
                    "reconstruction", images, global_step=epoch, dataformats="NHWC"
                )
                valid_writer.add_histogram(
                    "representation", representation, global_step=epoch
                )

            # Just get the loss for the rest, no summaries
            for (x,) in it:
                x = x.to(DEVICE)
                reconstruction, representation = model(x)
                loss = loss_fn(x, reconstruction)
                valid_losses.append(loss.item())
            valid_loss = np.mean(valid_losses)

            train_writer.add_scalar("loss", train_loss, global_step=epoch)
            valid_writer.add_scalar("loss", valid_loss, global_step=epoch)

            end_time = time.perf_counter() - start_time
            mean_time = train_time / len(train_dl)
            print(
                f"Epoch {epoch:03d}: {end_time:.2f}s ({mean_time:.2f}s per "
                f"batch), train loss: {train_loss:.4f}, valid loss: "
                f"{valid_loss:.4f}"
            )

        if epoch % save_every == 0:
            save_model(
                logs,
                epoch=epoch,
                model=model,
                model_args=model_args,
                optimiser=optimiser,
                optimiser_args=optimiser_args,
            )

    save_model(
        logs,
        epoch=final_epoch,
        model=model,
        model_args=model_args,
        optimiser=optimiser,
        optimiser_args=optimiser_args,
    )


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=Path)
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--batch_size", type=int, default=128)
def generate(input: Path, output: Path, model_path: Path, batch_size: int):
    if model_path.is_dir():
        save_path = get_latest_save(model_path)
    else:
        save_path = model_path
    print(f"Restoring model from {save_path}")
    load_dict = torch.load(save_path)
    model = TimeRecurrentAutoencoder(**load_dict["model_args"])
    model.cuda()
    model.load_state_dict(load_dict["model_state_dict"])

    x, names, corpus = get_data(input)

    representations = np.empty((len(x), model.embedding_size))
    with torch.no_grad():
        model.eval()
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i : i + batch_size], device=DEVICE)
            _, representation = model(batch)
            representations[i : i + batch_size, :] = representation.cpu()

    feature_names = [f"audeep{i + 1}" for i in range(representations.shape[-1])]
    write_features(
        output,
        corpus=corpus,
        names=names,
        features=representations,
        feature_names=feature_names,
    )
    print(f"Wrote netCDF4 file to {output}")


@click.group(no_args_is_help=True)
def main():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    # Workaround for having both PyTorch and TensorFlow installed.
    import tensorboard.compat.tensorflow_stub.io.gfile as _gfile
    import tensorflow as _tensorflow

    _tensorflow.io.gfile = _gfile


if __name__ == "__main__":
    main.add_command(generate)
    main.add_command(train)
    main()
