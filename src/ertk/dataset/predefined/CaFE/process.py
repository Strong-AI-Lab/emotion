"""Process the raw CaFE dataset.

This assumes the file structure from the original compressed file:
/.../
    Colère/
        Faible/
            *.aiff [for 192k data]
            *.wav  [for 48k data]
        Fort/
    ...
"""

from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "C": "anger",
    "D": "disgust",
    "J": "happiness",
    "N": "neutral",
    "P": "fear",
    "S": "surprise",
    "T": "sadness",
}

sentence_map = {
    "1": "Un cheval fou dans mon jardin",
    "2": "Deux ânes aigris au pelage brun",
    "3": "Trois cygnes aveugles au bord du lac",
    "4": "Quatre vieilles truies éléphantesques",
    "5": "Cinq pumas fiers et passionnés",
    "6": "Six ours aimants domestiqués",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the CaFE dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    if len(paths) == 0:
        paths = list(input_dir.glob("**/*.aiff"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")

    write_annotations({p.stem: emotion_map[p.stem[3]] for p in paths}, "label")
    speaker_dict = {p.stem: p.stem[:2] for p in paths}
    write_annotations(speaker_dict, "speaker")
    write_annotations(
        {k: ["F", "M"][int(v) % 2] for k, v in speaker_dict.items()},
        "gender",
    )
    sentence_dict = {p.stem: p.stem[-1] for p in paths}
    write_annotations(sentence_dict, "sentence")
    transcript = {k: sentence_map[v] for k, v in sentence_dict.items()}
    write_annotations(transcript, "transcript")
    write_annotations({p.stem: "fr" for p in paths}, "language")
    write_annotations({p.stem: "ca" for p in paths}, "country")


if __name__ == "__main__":
    main()
