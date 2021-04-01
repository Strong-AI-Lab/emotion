import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence, Tuple, Union

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from emorec.dataset import get_audio_paths, write_netcdf_dataset
from emorec.utils import PathlibPath

OPENSMILE_DIR = Path("third_party", "opensmile")
DEFAULT_CONF = OPENSMILE_DIR / "conf" / "IS09.conf"
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


def opensmile(
    path: Union[str, Path],
    config: Union[str, Path],
    tmp: Union[str, Path] = "tmp",
    debug: bool = False,
    restargs: Sequence[str] = [],
):
    name = Path(path).stem
    output_file = Path(tmp) / f"{name}.csv"
    if output_file.exists():
        output_file.unlink()

    smile_args = [
        OPENSMILE_BIN,
        "-C",
        str(config),
        "-I",
        str(path),
        "-csvoutput",
        str(output_file),
        "-classes",
        "{unknown}",
        "-class",
        "unknown",
        "-instname",
        name,
        *restargs,
    ]

    subprocess.run(
        smile_args,
        stdout=None if debug else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=False,
    )


def process_csv(path: Union[str, Path]):
    df = pd.read_csv(path, quotechar="'", header=None)
    return df.iloc[:, 1:]


@click.command()
@click.argument("corpus", type=str)
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option(
    "--config", type=Path, default=DEFAULT_CONF, help="Path to openSMILE config file."
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
    """Batch process a list of files in INPUT using the openSMILE
    Toolkit. The corpus name is set to CORPUS and a netCDF dataset is
    written to OUTPUT.
    """

    if not config.exists():
        raise FileNotFoundError("Config file doesn't exist")

    input_list = get_audio_paths(input)
    names = sorted(f.stem for f in input_list)

    tmp = tempfile.mkdtemp(prefix="opensmile_", suffix=f"_{corpus}")
    print(f"Using temp directory {tmp}")
    parallel_args = dict(n_jobs=1 if debug else -1, verbose=1)
    Parallel(prefer="threads", **parallel_args)(
        delayed(opensmile)(path, config, tmp, debug, smileargs) for path in input_list
    )

    tmp_files = [Path(tmp) / f"{name}.csv" for name in names]
    missing = [f for f in tmp_files if not f.exists()]
    if len(missing) > 0:
        raise RuntimeError(
            "Not all audio files were processed properly. These files are "
            "missing:\n" + "\n".join(map(str, missing))
        )
    # Use CPUs for this because I don't think it releases the GIL
    # for the whole processing.
    arr_list = Parallel(**parallel_args)(
        delayed(process_csv)(path) for path in tmp_files
    )
    shutil.rmtree(tmp)

    # This should be a 2D array
    full_array = np.concatenate(arr_list, axis=0)
    assert len(full_array.shape) == 2

    output.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(
        output,
        corpus=corpus,
        names=names,
        features=full_array,
        slices=[x.shape[0] for x in arr_list],
    )
    print(f"Wrote netCDF dataset to {output}")


if __name__ == "__main__":
    main()
