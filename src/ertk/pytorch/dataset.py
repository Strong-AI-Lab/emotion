"""PyTorch dataset utilities.

.. autosummary::
    :toctree:

    DatasetFuncWrapper
    DataModuleAdapter
    MTLDataModule
    MTLDataset
    create_mtl_dataloader
"""

from collections.abc import Callable, Collection
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset

from ertk.dataset import Dataset, load_datasets_config
from ertk.train import ExperimentConfig

__all__ = [
    "DatasetFuncWrapper",
    "DataModuleAdapter",
    "MTLDataModule",
    "MTLDataset",
    "create_mtl_dataloader",
]


class DatasetFuncWrapper(TorchDataset[tuple[torch.Tensor, ...]]):
    funcs: list[Callable[[torch.Tensor], torch.Tensor]]

    def __init__(self, dataset: TorchDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.funcs = []

    def add_func(self, func: Callable[[torch.Tensor], torch.Tensor]):
        self.funcs.append(func)

    def __getitem__(self, index: Any) -> tuple[torch.Tensor, ...]:
        x, *rest = self.dataset[index]
        for func in self.funcs:
            x = func(x)
        return (x,) + tuple(rest)

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        else:
            return NotImplemented


class DataModuleAdapter(pl.LightningDataModule):
    train_dataset: TorchDataset
    val_dataset: TorchDataset
    test_dataset: TorchDataset

    def __init__(
        self,
        *,
        load_config: Optional[ExperimentConfig] = None,
        dataset: Optional[Dataset] = None,
        train_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        val_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        test_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        batch_size: int = 32,
        dl_num_workers: int = 0,
    ):
        super().__init__()

        self.load_config = load_config
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_select = train_select
        self.val_select = val_select
        self.test_select = test_select if test_select is not None else val_select
        self.dl_num_workers = dl_num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if self.load_config:
            data = load_datasets_config(self.load_config)
            self.train_idx = data.get_idx_for_split(self.load_config.eval.train)
            self.val_idx = data.get_idx_for_split(self.load_config.eval.valid)
            self.test_idx = data.get_idx_for_split(self.load_config.eval.test)

        if isinstance(self.train_select, np.ndarray):
            self.train_idx = self.train_select
        else:
            self.train_idx = self.dataset.get_idx_for_split(self.train_select)
        if isinstance(self.val_select, np.ndarray):
            self.val_idx = self.val_select
        else:
            self.val_idx = self.dataset.get_idx_for_split(self.val_select)
        if isinstance(self.test_select, np.ndarray):
            self.test_idx = self.test_select
        else:
            self.test_idx = self.dataset.get_idx_for_split(self.test_select)

        self._create_datasets()

    def _create_datasets(self) -> None:
        train_x = torch.from_numpy(self.dataset.x[self.train_idx])
        train_y = torch.from_numpy(self.dataset.y[self.train_idx])
        self.train_dataset = TensorDataset(train_x, train_y)
        val_x = torch.from_numpy(self.dataset.x[self.val_idx])
        val_y = torch.from_numpy(self.dataset.y[self.val_idx])
        self.val_dataset = TensorDataset(val_x, val_y)
        test_x = torch.from_numpy(self.dataset.x[self.test_idx])
        test_y = torch.from_numpy(self.dataset.y[self.test_idx])
        self.test_dataset = TensorDataset(test_x, test_y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dl_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_num_workers,
        )


class MTLDataModule(DataModuleAdapter):
    def __init__(
        self,
        *,
        dataset: Dataset,
        tasks: list[str],
        train_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        val_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        test_select: Union[str, dict[str, Collection[str]], np.ndarray, None] = None,
        batch_size: int = 32,
        dl_num_workers: int = 0,
    ):
        super().__init__(
            dataset=dataset,
            train_select=train_select,
            val_select=val_select,
            test_select=test_select,
            batch_size=batch_size,
            dl_num_workers=dl_num_workers,
        )
        self.tasks = tasks

    def _create_datasets(self) -> None:
        for name in ["train", "val", "test"]:
            idx = getattr(self, f"{name}_idx")
            x = torch.as_tensor(self.dataset.x[idx])
            ys = {
                k: torch.as_tensor(
                    self.dataset.get_group_indices(k)[idx], dtype=torch.long
                )
                for k in self.tasks
            }
            setattr(self, f"{name}_dataset", MTLDataset(x, ys))


class MTLDataset(TorchDataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, x: torch.Tensor, y: dict[str, torch.Tensor]):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], {k: v[index] for k, v in self.y.items()}

    def __len__(self):
        return self.x.size(0)


def create_mtl_dataloader(
    data: Dataset, tasks: list[str], batch_size: int = 32, shuffle: bool = False
) -> DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    x = torch.as_tensor(data.x)
    task_data = {k: torch.as_tensor(data.get_group_indices(k)) for k in tasks}
    dataset = MTLDataset(x, task_data)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
