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
@click.argument("corpus", type=str)
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.argument("output", type=Path)
@click.option(
    "--checkpoint",
    type=PathlibPath(exists=True, dir_okay=False),
    required=True,
    help="Path to checkpoint.",
)
@click.option("--type", "tp", type=int, default=1, help="Wav2Vec version, 1 or 2.")
def main(corpus: str, input: Path, output: Path, checkpoint: Path, tp: int):
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
    filepaths = get_audio_paths(input)
    names = [x.stem for x in filepaths]
    for filepath in tqdm(filepaths):
        wav, _ = librosa.load(filepath, sr=16000, mono=True, res_type="kaiser_fast")
        tensor = torch.tensor(wav, device="cuda").unsqueeze(0)
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
        output, names=names, features=embeddings, corpus=corpus
    )
    print(f"Wrote netCDF4 dataset to {output}")


if __name__ == "__main__":
    main()
