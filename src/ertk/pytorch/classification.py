import logging
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset

from ertk.pytorch.dataset import MTLDataModule
from ertk.pytorch.models.mtl import MTLModel
from ertk.pytorch.utils import LightningWrapper
from ertk.train import get_scores
from ertk.utils import ScoreFunction


def pt_cross_validate(
    model_fn: Callable[..., Union[nn.Module, pl.LightningModule]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    cv: BaseCrossValidator = LeaveOneGroupOut(),
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    verbose: int = 0,
    fit_params: Dict[str, Any] = {},
):
    batch_size = fit_params.pop("batch_size")
    sw = fit_params.pop("sample_weight", None)

    logging.debug(f"cv={cv}")
    logging.debug(f"sample_weight={sw}")
    logging.debug(f"fit_params={fit_params}")

    n_folds = cv.get_n_splits(x, y, groups)
    scores = defaultdict(list)
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        if verbose:
            logging.info(f"Fold {fold + 1}/{n_folds}")

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        sw_train = sw[train] if sw is not None else None
        sw_test = sw[test] if sw is not None else None

        _scores = pt_train_val_test(
            model_fn,
            train_data=(x_train, y_train, sw_train),
            valid_data=(x_test, y_test, sw_test),
            test_data=(x_test, y_test, sw_test),
            scoring=scoring,
            verbose=verbose,
            fit_params=dict(fit_params),
            batch_size=batch_size,
        )

        for k in _scores:
            scores[k].append(_scores[k])
    return {k: np.array(scores[k]) for k in scores}


def pt_train_val_test(
    model_fn: Callable[..., Union[nn.Module, pl.LightningModule]],
    train_data: Tuple[np.ndarray, ...],
    valid_data: Tuple[np.ndarray, ...],
    test_data: Optional[Tuple[np.ndarray, ...]] = None,
    scoring: Union[
        str, List[str], Dict[str, ScoreFunction], Callable[..., float]
    ] = "accuracy",
    verbose: int = 0,
    batch_size: int = 32,
    fit_params: Dict[str, Any] = None,
):
    fit_params = fit_params or {}
    log_dir = fit_params.pop("log_dir", None)
    batch_size = fit_params.pop("batch_size", batch_size)
    n_gpus = fit_params.pop("n_gpus", 1)

    logging.debug(f"log_dir={log_dir}")

    logger: Union[bool, List] = False
    if log_dir is not None:
        logger = [
            TensorBoardLogger(log_dir, name=None, version=""),
            CSVLogger(log_dir, name=None, version=""),
        ]

    if test_data is None:
        test_data = valid_data

    clf = model_fn()
    logging.debug(clf)
    if isinstance(clf, tuple):
        transform = clf[0]
        clf = clf[1]
        transform.fit(train_data[0], y=train_data[1])
        train_data = (transform.transform(train_data[0]), *train_data[1:])
        valid_data = (transform.transform(valid_data[0]), *valid_data[1:])
        test_data = (transform.transform(test_data[0]), *test_data[1:])

    if not isinstance(clf, pl.LightningModule):
        clf = LightningWrapper(
            clf,
            nn.CrossEntropyLoss(reduction="none"),
            optim_fn=fit_params.pop("optim_fn"),
        )

    train_data_pt = tuple(torch.from_numpy(x) for x in train_data)
    valid_data_pt = tuple(torch.from_numpy(x) for x in valid_data)
    test_data_pt = tuple(torch.from_numpy(x) for x in test_data)

    train_ds = TensorDataset(*train_data_pt)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_ds = TensorDataset(*valid_data_pt)
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_ds = TensorDataset(*test_data_pt)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    if verbose <= 0:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.simplefilter("ignore", UserWarning)
    trainer = pl.Trainer(
        gpus=n_gpus,
        strategy="dp",
        max_epochs=fit_params.pop("max_epochs"),
        logger=logger,
        enable_progress_bar=bool(verbose),
        enable_model_summary=bool(verbose),
        enable_checkpointing=False,
    )
    trainer.fit(clf, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    y_pred = torch.argmax(torch.cat(trainer.predict(clf, test_dl)), -1).cpu().numpy()
    warnings.simplefilter("default", UserWarning)

    y_true = test_data[1]
    scores = get_scores(scoring, y_pred, y_true)
    scores = {f"test_{k}": v for k, v in scores.items()}
    return scores


def train_mtl_model(
    model: MTLModel,
    tasks: Dict[int, int],
    data: Optional[MTLDataModule],
    train_data: Optional[DataLoader] = None,
    valid_data: Optional[DataLoader] = None,
    epochs: int = 200,
    device: str = "cuda:0",
    verbose: int = 0,
):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        logger=bool(verbose),
        enable_progress_bar=bool(verbose),
        enable_model_summary=bool(verbose),
        enable_checkpointing=False,
    )
    cols = [f"{x}_acc" for x in tasks] + [f"{x}_loss" for x in tasks]
    stats_df = pd.DataFrame(
        columns=pd.Index(cols, name="metric"),
        index=pd.MultiIndex.from_product(
            [range(epochs), ["train", "val"]], names=["epoch", "phase"]
        ),
    )

    if data is not None:
        trainer.fit(model, data)
        [val] = trainer.validate(model, data, verbose=verbose)
    else:
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)
        [val] = trainer.validate(model, valid_data, verbose=verbose)

    return val
