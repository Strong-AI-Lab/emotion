from pathlib import Path

import click
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ertk.config import get_arg_mapping
from ertk.pytorch import get_pt_model
from ertk.pytorch.utils import LightningWrapper


@click.command()
@click.argument("model_name")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
@click.option("--1D/--2D", "is_1d", default=True, help="Input type.")
@click.option(
    "--summary_only", is_flag=True, help="Print summary but don't train on dummy data."
)
@click.option(
    "--memory/--generated",
    default=False,
    help="Have all data in memory or generate batches.",
)
@click.option("--model_args", type=click.Path(exists=True, path_type=Path))
@click.option("--train", type=int, default=8000, help="Number of training instances.")
@click.option("--valid", type=int, default=800, help="Number of validation instances.")
@click.option("--features", type=int, default=1024, help="Dimensionality of input.")
@click.option("--steps", type=int, default=512, help="Length of sequence.")
def test_fit(
    model_name: str,
    batch_size: int,
    is_1d: bool,
    summary_only: bool,
    memory: bool,
    model_args: Path,
    train: int,
    valid: int,
    features: int,
    steps: int,
):
    """Trains a model for 2 epochs with random data and prints timing
    information for testing purposes. The default number of instances is
    8000, which is larger than the current largest dataset, MSP-IMPROV.
    """

    args = get_arg_mapping(model_args) if model_args else {}
    model = get_pt_model(model_name, n_features=features, n_classes=4, **args)
    if not isinstance(model, pl.LightningModule):
        model = LightningWrapper(model, nn.CrossEntropyLoss(), lambda x: Adam(x, 0.001))
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable parameters: {}".format(trainable_params))
    print("Total parameters: {}".format(total_params))
    print()

    if summary_only:
        return

    shape = (features,) if is_1d else (steps, features)

    rng = np.random.default_rng()
    train_x = torch.as_tensor(rng.random((train,) + shape, dtype=np.float32))
    train_y = torch.as_tensor(rng.integers(4, size=train))
    valid_x = torch.as_tensor(rng.random((valid,) + shape, dtype=np.float32))
    valid_y = torch.as_tensor(rng.integers(4, size=valid))

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(valid_x, valid_y)
    train_dl = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_dl = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=2,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    preds = torch.cat(trainer.predict(model, valid_dl)).cpu().numpy()
    print(preds)


if __name__ == "__main__":
    test_fit()
