from pathlib import Path

import click
import numpy as np
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecModel
from tqdm import tqdm

from emorec.dataset import Dataset, write_features
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option(
    "--checkpoint",
    type=PathlibPath(exists=True, dir_okay=False),
    required=True,
    help="Path to checkpoint.",
)
@click.option("--type", "tp", type=int, default=1, help="Wav2Vec version, 1 or 2.")
def main(input: Path, output: Path, checkpoint: Path, tp: int):
    dataset = Dataset(input)

    print(f"Loading model from {checkpoint}")
    cp = torch.load(checkpoint)
    if tp == 1:
        model = Wav2VecModel.build_model(cp["args"], task=None)
    else:
        model = Wav2Vec2Model.build_model(cp["args"], task=None)
    model.load_state_dict(cp["model"])
    model.cuda()
    model.eval()

    _embeddings = []
    for wav in tqdm(dataset.x):
        # Transpose so that the single 'feature' dimension becomes 'batch' dimension
        tensor = torch.tensor(wav.T, device="cuda")
        with torch.no_grad():
            if tp == 1:
                z = model.feature_extractor(tensor)
                if model.vector_quantizer is not None:
                    z, _ = model.vector_quantizer.forward_idx(z)
                c = model.feature_aggregator(z)
            else:
                c, _ = model.extract_features(tensor, None)
                c = c.transpose(-2, -1)
            _embeddings.append(c[0].mean(-1).cpu().numpy())
    embeddings = np.stack(_embeddings)

    write_features(
        output, names=dataset.names, features=embeddings, corpus=dataset.corpus
    )
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
