from pathlib import Path

import click
import numpy as np
import librosa
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecModel
from tqdm import tqdm

from ertk.dataset import write_features, get_audio_paths
from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option("--corpus", required=True)
@click.option(
    "--checkpoint",
    type=PathlibPath(exists=True, dir_okay=False),
    required=True,
    help="Path to checkpoint.",
)
@click.option("--type", "tp", type=int, default=1, help="Wav2Vec version, 1 or 2.")
@click.option(
    "--aggregate",
    type=click.Choice(["mean", "max", "none"]),
    default="mean",
    help="Whether and how to pool embeddings across time.",
)
@click.option(
    "--context/--encoder",
    default=True,
    help="Use context embeddings or encoder embeddings (default: context).",
)
def main(
    corpus: str,
    input: Path,
    output: Path,
    checkpoint: Path,
    tp: int,
    aggregate: str,
    context: bool,
):
    """Extracts wav2vec embeddings for given INPUT audio clips, and
    writes features to OUTPUT. This script can extract encoder or
    context embeddings, and can produce the sequence of vectors or pool
    over time to produce an aggregated vector.
    """
    print(f"Loading model from {checkpoint}")
    cp = torch.load(checkpoint)
    if tp == 1:
        model = Wav2VecModel.build_model(cp["args"], task=None)
    else:
        model = Wav2Vec2Model.build_model(cp["args"], task=None)
    model.load_state_dict(cp["model"])
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    _embeddings = []
    filepaths = get_audio_paths(input)
    names = [x.stem for x in filepaths]
    for filepath in tqdm(filepaths, disable=None):
        wav, _ = librosa.load(filepath, sr=16000, mono=True, res_type="kaiser_fast")
        tensor = torch.tensor(wav, device="cuda").unsqueeze(0)
        if tp == 1:  # Original wav2vec
            z = model.feature_extractor(tensor)
            if model.vector_quantizer is not None:
                z, _ = model.vector_quantizer.forward_idx(z)
            feats = model.feature_aggregator(z) if context else z
        else:  # wav2vec 2.0
            if context:
                c, _ = model.extract_features(tensor, None)
                # Transpose to (batch, feats, steps)
                feats = c.transpose(-2, -1)
            else:
                feats = model.feature_extractor(tensor)

        if aggregate == "none":
            _embeddings.append(feats[0].transpose(1, 0).cpu().numpy())
        elif aggregate == "mean":
            _embeddings.append(feats[0].mean(-1).cpu().numpy())
        elif aggregate == "max":
            _embeddings.append(feats[0].max(-1).cpu().numpy())
    if aggregate == "none":
        slices = [len(x) for x in _embeddings]
        embeddings = np.concatenate(_embeddings)
    else:
        embeddings = np.stack(_embeddings)
        slices = np.ones(len(embeddings))

    ft = "context" if context else "encoder"
    feature_names = [f"wav2vec_{ft}_{i}" for i in range(embeddings.shape[-1])]
    write_features(
        output,
        names=names,
        features=embeddings,
        slices=slices,
        corpus=corpus,
        feature_names=feature_names,
    )
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
