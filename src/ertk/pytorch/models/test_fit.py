from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ertk.pytorch import get_pt_model
from ertk.pytorch.models import PyTorchModelConfig


class RandomDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self, total: int, shapes: list[tuple[int]], dtypes: list[torch.dtype]
    ) -> None:
        super().__init__()
        self.total = total
        self.data = []
        for shape, dtype in zip(shapes, dtypes):
            self.data.append(torch.zeros(shape, dtype=dtype))

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        return tuple(self.data)

    def __len__(self) -> int:
        return self.total


@click.command()
@click.argument("model_name")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
@click.option("--shape", default="[(1024,), ()]", help="Shapes.")
@click.option("--dtype", default="[torch.float32, torch.int64]", help="Dtypes.")
@click.option(
    "--summary_only", is_flag=True, help="Print summary but don't train on dummy data."
)
@click.option("--model_config", type=click.Path(exists=True, path_type=Path))
@click.option("--train", type=int, default=8000, help="Number of training instances.")
@click.option("--valid", type=int, default=800, help="Number of validation instances.")
@click.argument("restargs", nargs=-1)
def test_fit(
    model_name: str,
    batch_size: int,
    shape: str,
    dtype: str,
    summary_only: bool,
    model_config: Path,
    train: int,
    valid: int,
    restargs: tuple[str],
):
    """Trains a model for 2 epochs with random data and prints timing
    information for testing purposes.
    """

    config: PyTorchModelConfig
    if model_config:
        config = OmegaConf.load(model_config)
    else:
        config = PyTorchModelConfig(
            optimiser="adam", learning_rate=0.001, loss="cross_entropy"
        )

    cli_args = OmegaConf.from_cli(list(restargs))
    if cli_args.model_config:
        config = OmegaConf.merge(config, cli_args.model_config)

    model = get_pt_model(model_name, config=config)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable parameters: {}".format(trainable_params))
    print("Total parameters: {}".format(total_params))
    print()

    if summary_only:
        return

    shapes = eval(shape, {})
    dtypes = eval(dtype, {"torch": torch})
    train_data = RandomDataset(train, shapes, dtypes)
    valid_data = RandomDataset(valid, shapes, dtypes)
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
    trainer.predict(model, valid_dl)


if __name__ == "__main__":
    test_fit()
