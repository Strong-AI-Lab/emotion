import joblib
import tqdm


# Class adapted from user394430's answer here:
# https://stackoverflow.com/a/61900501/10044861
# Licensed under CC BY-SA 4.0
class TqdmParallel(joblib.Parallel):
    """Convenience class that acts identically to joblib.Parallel except
    it uses a tqdm progress bar.
    """

    def __init__(
        self,
        total: int = 1,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
        **kwargs,
    ):
        self.total = total
        self.tqdm_args = {"desc": desc, "unit": unit, "leave": leave, "disable": None}
        kwargs["verbose"] = 0
        super().__init__(**kwargs)

    def __call__(self, iterable):
        with tqdm.tqdm(total=self.total, **self.tqdm_args) as self.pbar:
            return super().__call__(iterable)

    def print_progress(self):
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()
