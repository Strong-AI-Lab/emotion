import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
from joblib import delayed

from ertk.dataset import get_audio_paths, write_features
from ertk.utils import PathlibPath, TqdmParallel

OPENSMILE_DIR = Path("third_party", "opensmile")
OPENSMILE_BIN = "SMILExtract"
try:
    subprocess.check_call(
        [OPENSMILE_BIN, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
except FileNotFoundError:
    OPENSMILE_BIN = str(OPENSMILE_DIR / "SMILExtract")
    if sys.platform == "win32":
        OPENSMILE_BIN = str(OPENSMILE_DIR / "SMILExtract.exe")
    subprocess.check_call(
        [OPENSMILE_BIN, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )


@click.command()
@click.argument("corpus", type=str)
@click.argument("input", type=PathlibPath(exists=True))
@click.argument("output", type=Path)
@click.option(
    "--config",
    type=PathlibPath(exists=True, dir_okay=False),
    required=True,
    help="Path to openSMILE config file.",
)
@click.option(
    "--debug", is_flag=True, help="Disable multiprocessing to highlight errors."
)
@click.argument("smileargs", nargs=-1)
def main(
    corpus: str,
    input: Path,
    output: Path,
    config: Path,
    debug: bool,
    smileargs: Tuple[str],
):
    """Process a list of files in INPUT using the openSMILE
    Toolkit. The corpus name is set to CORPUS and a netCDF dataset is
    written to OUTPUT.
    """

    input_list = sorted(get_audio_paths(input))
    tmp = tempfile.mkdtemp(prefix="opensmile_", suffix=f"_{corpus}")
    tmp_files = [Path(tmp, f"{path.stem}.csv") for path in input_list]
    n_jobs = 1 if debug else -1

    print(
        f"Using SMILExtract at {OPENSMILE_BIN} with extra options {' '.join(smileargs)}"
    )
    TqdmParallel(total=len(input_list), desc="Processing files", n_jobs=n_jobs)(
        delayed(subprocess.run)(
            [
                OPENSMILE_BIN,
                "-C",
                str(config),
                "-I",
                str(path),
                "-csvoutput",
                str(tmp_file),
                "-instname",
                path.stem,
                *smileargs,
            ],
            stdout=None if debug else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        for path, tmp_file in zip(input_list, tmp_files)
    )
    missing = {f.stem for f in tmp_files if not f.exists()}
    if len(missing) > 0:
        raise RuntimeError(
            "Not all audio files were processed properly. These names are "
            "missing:\n" + "\n\t".join(missing)
        )
    df_list = TqdmParallel(total=len(tmp_files), desc="Processing CSVs", n_jobs=n_jobs)(
        delayed(pd.read_csv)(path, quotechar="'", header=0, index_col=0)
        for path in tmp_files
    )
    shutil.rmtree(tmp, ignore_errors=True)
    feats = np.concatenate(df_list, axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_features(
        output,
        corpus=corpus,
        names=[x.stem for x in input_list],
        features=feats,
        feature_names=list(df_list[0].columns),
        slices=[len(x) for x in df_list],
    )
    print(f"Wrote dataset to {output}")


if __name__ == "__main__":
    main()
