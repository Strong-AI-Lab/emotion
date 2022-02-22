from typing import Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ertk.dataset import Dataset
from ertk.pytorch.utils import MTLTaskConfig


class DataModuleAdapter(pl.LightningDataModule):
    train_dataset: TorchDataset
    val_dataset: TorchDataset
    test_dataset: TorchDataset

    def __init__(
        self,
        dataset: Dataset,
        train_select: Union[str, Dict[str, str]],
        val_select: Union[str, Dict[str, str]],
        test_select: Union[str, Dict[str, str], None] = None,
        batch_size: int = 32,
        dl_num_workers: int = 0,
    ):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.train_select = train_select
        self.val_select = val_select
        self.test_select = test_select or val_select
        self.dl_num_workers = dl_num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_idx = self.dataset.get_idx_for_split(self.train_select)
        self.val_idx = self.dataset.get_idx_for_split(self.val_select)
        self.test_idx = self.dataset.get_idx_for_split(self.test_select)

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
        dataset: Dataset,
        tasks: Dict[str, MTLTaskConfig],
        train_select: Union[str, Dict[str, str]],
        val_select: Union[str, Dict[str, str]],
        test_select: Union[str, Dict[str, str], None] = None,
        batch_size: int = 32,
        dl_num_workers: int = 0,
    ):
        super().__init__(
            dataset,
            train_select=train_select,
            val_select=val_select,
            test_select=test_select,
            batch_size=batch_size,
            dl_num_workers=dl_num_workers,
        )
        self.tasks = tasks

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

        for name in ["train", "val", "test"]:
            idx = getattr(self, f"{name}_idx")
            x = torch.as_tensor(self.dataset.x[idx])
            ys = {
                k: torch.as_tensor(self.dataset.get_group_indices(k)[idx])
                for k in self.tasks
            }
            setattr(self, f"{name}_dataset", MTLDataset(x, ys))


class MTLDataset(TorchDataset[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    def __init__(self, x: torch.Tensor, y: Dict[str, torch.Tensor]):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], {k: v[index] for k, v in self.y.items()}

    def __len__(self):
        return self.x.size(0)


def create_mtl_dataloader(
    data: Dataset,
    tasks: Dict[str, MTLTaskConfig],
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    x = torch.as_tensor(data.x)
    task_data = {k: torch.as_tensor(data.get_group_indices(k)) for k in tasks}
    dataset = MTLDataset(x, task_data)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
