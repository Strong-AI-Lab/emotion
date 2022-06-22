import multiprocessing
import os
from functools import partial
from typing import Iterable

import joblib
import tqdm


# Class adapted from user394430's answer here:
# https://stackoverflow.com/a/61900501/10044861
# Licensed under CC BY-SA 4.0
class TqdmParallel(joblib.Parallel):
    """Convenience class that acts identically to joblib.Parallel except
    it uses a tqdm progress bar.

    Parameters
    ----------
    total: int
        Total number of items in the iterable.
    desc: str
        Progress bar description.
    unit: str
        Progress bar unit.
    leave: bool
        Whether to leave the progress bar after completion.
    **kwargs: dict
        Other keyword args passed to joblib.Parallel.
    """

    def __init__(
        self,
        total: int = 1,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
        **kwargs,
    ):
        self.tqdm_args = {
            "total": total,
            "desc": desc,
            "unit": unit,
            "leave": leave,
            "disable": None,
        }
        kwargs["verbose"] = 0
        super().__init__(**kwargs)

    def __call__(self, iterable):
        with tqdm.tqdm(**self.tqdm_args) as self.pbar:
            return super().__call__(iterable)

    def print_progress(self):
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()


class TqdmMultiprocessing:
    def __init__(
        self,
        total: int = 1,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
        **kwargs,
    ) -> None:
        self.tqdm_args = {
            "total": total,
            "desc": desc,
            "unit": unit,
            "leave": leave,
            "disable": None,
            **kwargs,
        }

    def imap(self, func, iterable, *args, n_jobs=1, chunksize=1, **kwargs) -> Iterable:
        _f = partial(func, *args, **kwargs)
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        self.pbar = tqdm.tqdm(iterable, **self.tqdm_args)
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as pool:
                yield from pool.imap(_f, iter(self.pbar), chunksize=chunksize)
        else:
            yield from map(_f, iter(self.pbar))
        self.pbar.close()
