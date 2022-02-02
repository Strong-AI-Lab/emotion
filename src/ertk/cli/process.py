import logging
import pprint
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import click
from joblib import delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from ertk.dataset import get_audio_paths, read_features, write_features
from ertk.preprocessing import InstanceProcessor
from ertk.utils import TqdmParallel

_help_text = f"""Process data or audio files in INPUT, write features to
OUTPUT. The available feature extractors are:

{InstanceProcessor.valid_preprocessors()}
"""


def print_features_help(ctx, param, val):
    if not val or ctx.resilient_parsing:
        return
    print(
        "Default configurations for registered feature extractors. Items with '???' "
        "are required to be specified manually.\n"
    )
    for key in InstanceProcessor.valid_preprocessors():
        print(f"Args for '{key}':")
        config_cls = InstanceProcessor.get_processor_class(key).get_config_type()
        for k, v in asdict(config_cls()).items():
            print(f"  {k} = {v}")
        print()
    ctx.exit()


@click.command(help=_help_text)
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=Path)
@click.option("--features", required=True, help="Features to extract.")
@click.option(
    "--features_help",
    is_flag=True,
    callback=print_features_help,
    expose_value=False,
    is_eager=True,
    help="Show options for registered feature extractors.",
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
    "--sample_rate", type=float, help="Resample to this rate for audio input."
)
@click.option("--n_jobs", type=int, default=-1, help="Number of parallel jobs to run.")
@click.option("--verbose", is_flag=True)
@click.argument("restargs", nargs=-1)
def process_main(
    input: Path,
    output: Path,
    features: str,
    config: Path,
    corpus: str,
    batch_size: int,
    sample_rate: float,
    n_jobs: int,
    verbose: bool,
    restargs: Tuple[str],
):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    extractor_cls = InstanceProcessor.get_processor_class(features)
    config_cls = extractor_cls.get_config_type()
    if config:
        conf = OmegaConf.load(config)
    else:
        conf = extractor_cls.get_default_config()
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(restargs)))
    if verbose:
        print("Using configuration:")
        pprint.pprint(dict(conf))
    extractor = extractor_cls(config_cls.from_config(conf))

    if input.suffix == ".txt":
        # TODO: unify with read_features()
        paths = get_audio_paths(input)
        names = [Path(x).stem for x in paths]
        if batch_size != 1:
            feats = list(
                tqdm(
                    extractor.process_files(
                        paths, batch_size=batch_size, sr=sample_rate
                    ),
                    total=len(paths),
                    desc="Processing files",
                )
            )
        else:
            feats = TqdmParallel(len(paths), "Processing files", n_jobs=n_jobs)(
                delayed(extractor.process_file)(x, sr=sample_rate) for x in paths
            )
    else:
        input_data = read_features(input)
        corpus = corpus or input_data.corpus
        names = input_data.names
        if batch_size != 1:
            feats = list(
                tqdm(
                    extractor.process_all(input_data.features, batch_size),
                    total=len(input_data),
                    desc="Processing files",
                )
            )
        else:
            feats = TqdmParallel(len(input_data), "Processing files", n_jobs=n_jobs)(
                map(delayed(extractor.process_instance), input_data.features)
            )
    write_features(output, feats, names=names, corpus=corpus)
    print(f"Wrote features to {output}")
