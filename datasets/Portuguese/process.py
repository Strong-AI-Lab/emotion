"""Process the raw Portuguese dataset (Castro & Lima).

This assumes the file structure from the original compressed file:
/.../
    Castro_2010_AppxB_Sents.txt
    Castro_2010_AppxB_Pseudosents.txt
    *.wav
"""

import re
from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist

REGEX = re.compile(r"^\d+[sp][AB]_([a-z]+)\d+$")

emotion_map = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "neutral": "neutral",
    "surprise": "surprise",
}

sentence_map = {
    # Sentences
    "estaMesa": "Esta mesa é de madeira",
    "oRadio": "O rádio está ligado",
    "aqueleLivro": "Aquele livro é de história",
    "aTerra": "A Terra é um planeta",
    "oCao": "O cão trouxe a bola",
    "eleChega": "Ele chega amanhã",
    "estaRoupa": "Esta roupa é colorida",
    "osJardins": "Os jardins têm flores",
    "asPessoas": "As pessoas vão a concertos",
    "haArvores": "Há árvores na floresta",
    "osTigres": "Os tigres são selvagens",
    "oQuadro": "O quadro está na parede",
    "alguemFechou": "Alguém fechou as janelas",
    "osJovens": "Os jovens ouvem música",
    "oFutebol": "O futebol é um desporto",
    "elaViajou": "Ela viajou de comboio",
    # Pseudo-sentences
    "estaDepa": "Esta dêpa é de faneira",
    "oDarrio": "O dárrio está guilado",
    "aqueleJicro": "Aquele jicro é de hisbólia",
    "aPirra": "A Pirra é um flaneto",
    "oLao": "O lão droube a nóma",
    "eleChena": "Ele chena aguinhã",
    "estaSouda": "Esta souda é lacoripa",
    "osBartins": "Os bartins têm pléres",
    "asSemoas": "As semoas vão a cambêrtos",
    "haArjuques": "Há árjuques na plurisca",
    "osLagres": "Os lagres são siltávens",
    "oJuadre": "O juadre está na pafêne",
    "alguemBelhou": "Alguém belhou as jalétas",
    "osDofens": "Os dófens mavem tézica",
    "oDutebel": "O dutebel é um nesforpo",
    "elaJiavou": "Ela jiavou de lantóio",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the Portuguese dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")

    sent_info = pd.concat(
        [
            pd.read_csv(
                input_dir / x,
                sep="\t",
                skiprows=5,
                header=0,
                index_col=0,
            )
            for x in [
                "Castro_2010_AppxB_Sents.txt",
                "Castro_2010_AppxB_Pseudosents.txt",
            ]
        ],
        ignore_index=True,
    ).set_index("Stimulus ")

    write_annotations(sent_info["Content"].map(sentence_map), "transcript")
    write_annotations(sent_info["Intensity (1-7)"], "intensity")
    write_annotations(
        {p.stem: emotion_map[REGEX.match(p.stem).group(1)] for p in paths},
        "label",
    )
    write_annotations({p.stem: p.stem[p.stem.find("_") - 1] for p in paths}, "speaker")
    write_annotations(
        {p.stem: p.stem[p.stem.find("_") - 2] for p in paths}, "sent_type"
    )
    write_annotations({p.stem: "pt" for p in paths}, "language")


if __name__ == "__main__":
    main()
