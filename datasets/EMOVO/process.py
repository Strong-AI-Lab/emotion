"""Process the raw EMOVO dataset.

This assumes the file structure from the original compressed file:
/.../
    f1/
        *.wav
    m1/
        *.wav
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "dis": "disgust",
    "gio": "happiness",
    "neu": "neutral",
    "pau": "fear",
    "rab": "anger",
    "sor": "surprise",
    "tri": "sadness",
}

sentence_map = {
    "b1": "Gli operai si alzano presto",
    "b2": "I vigili sono muniti di pistola",
    "b3": "La cascata fa molto rumore",
    "l1": "L'autunno prossimo Tony partirà per la Spagna nella prima metà di ottobre",
    "l2": "Ora prendo la felpa di là ed esco per fare una passeggiata",
    "l3": "Un attimo dopo s'è incamminato ... ed è inciampato",
    "l4": "Vorrei il numero telefonico del Signor Piatti",
    "n1": "La casa forte vuole col pane",
    "n2": "La forza trova il passo e l'aglio rosso",
    "n3": "Il gatto sta scorrendo nella pera",
    "n4": "Insalata pastasciutta coscia d'agnello limoncello",
    "n5": "Uno quarantatré dieci mille cinquantasette venti",
    "d1": "Sabato sera cosa farà?",
    "d2": "Porti con te quella cosa?",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EMOVO dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("**/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    write_filelist(resample_dir.glob("*.wav"), "files_all")

    write_annotations({p.stem: emotion_map[p.stem[0:3]] for p in paths}, "label")
    write_annotations({p.stem: p.stem[4:6] for p in paths}, "speaker")
    write_annotations({p.stem: p.stem[4].upper() for p in paths}, "gender")
    write_annotations({p.stem: p.stem[7:9] for p in paths}, "sentence")
    write_annotations({p.stem: sentence_map[p.stem[7:9]] for p in paths}, "transcript")
    write_annotations({p.stem: "it" for p in paths}, "language")
    write_annotations({p.stem: "it" for p in paths}, "country")


if __name__ == "__main__":
    main()
