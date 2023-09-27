import logging
import pprint
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Tuple

import click
from omegaconf import OmegaConf
from tqdm import tqdm

from ertk.dataset import read_features_iterable, write_features
from ertk.utils import TqdmMultiprocessing

logger = logging.getLogger(__name__)


def list_processors(ctx: click.Context, param, val):
    from ertk.preprocessing import InstanceProcessor

    if not val or ctx.resilient_parsing:
        return

    print(f"\nAvailable processors:\n{InstanceProcessor.valid_processors()}\n")
    print(
        "Below are the default configurations for registered feature extractors. Items "
        "with '???' are required to be specified manually.\n"
    )
    for key in sorted(InstanceProcessor.valid_processors()):
        print(f"[{key}]: ", end="")
        proc_cls = InstanceProcessor.get_processor_class(key)
        print(", ".join([b.__name__ for b in proc_cls.__bases__]))
        config_cls = proc_cls.get_config_type()
        for k, v in asdict(config_cls()).items():
            print(f"  {k} = {v}")
        print()

    ctx.exit()


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=Path)
@click.option("--processor", "--features", required=True, help="Processor to apply.")
@click.option(
    "--list_processors",
    is_flag=True,
    callback=list_processors,
    expose_value=False,
    is_eager=True,
    help="Show options for registered processors.",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Extractor config file.",
)
@click.option("--corpus", default="", help="Corpus name to set.")
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for processing. If batch_size is greater than 1, clips will be "
    "batched together.",
)
@click.option(
    "--sample_rate",
    type=int,
    default=16000,
    help="Resample to this rate for audio input.",
)
@click.option("--n_jobs", type=int, default=1, help="Number of parallel jobs to run.")
@click.option("--verbose", type=int, default=0)
@click.argument("restargs", nargs=-1)
def main(
    input: Path,
    output: Path,
    processor: str,
    config: Path,
    corpus: str,
    batch_size: int,
    sample_rate: int,
    n_jobs: int,
    verbose: int,
    restargs: Tuple[str],
):
    """Process features or audio files in INPUT, write to OUTPUT."""

    from ertk.preprocessing import InstanceProcessor

    if verbose > 0:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    extractor_cls = InstanceProcessor.get_processor_class(processor)
    config_cls = extractor_cls.get_config_type()
    conf = (
        config_cls.from_file(config) if config else extractor_cls.get_default_config()
    )
    conf = config_cls.merge_with_args(conf, restargs)
    logger.info("Using configuration:")
    logger.info(pprint.pformat(config_cls.to_dictconfig(conf)))
    extractor = extractor_cls(config_cls.from_config(conf))

    input_data = read_features_iterable(input, sample_rate=sample_rate)
    corpus = corpus or input_data.corpus
    names = input_data.names
    if batch_size != 1:
        feats = tqdm(
            extractor.process_all(iter(input_data), batch_size, sr=sample_rate),
            total=len(input_data),
            desc="Processing files",
            disable=None,
        )
    else:
        # TODO: Set back to joblib when
        # https://github.com/joblib/joblib/pull/588 is merged
        pool = TqdmMultiprocessing(len(input_data), "Processing files", n_jobs=n_jobs)
        f = partial(extractor.process_instance, sr=sample_rate)
        feats = pool.imap(f, input_data)
    write_features(
        output, feats, names=names, corpus=corpus, feature_names=extractor.feature_names
    )
    extractor.finish()
    print(f"Wrote features to {output}")
