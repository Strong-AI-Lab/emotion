"""Process the raw EmoFilm dataset.

This assumes the file structure from the original compressed file:
/.../
    *.wav
    f_m_corpus_it_es_en_legend.xlsx
"""

from pathlib import Path

import click
import pandas as pd

from ertk.dataset import resample_audio, write_annotations, write_filelist

emotion_map = {
    "ans": "fear",
    "dis": "disgust",
    "gio": "happiness",
    "rab": "anger",
    "tri": "sadness",
}

film_map = {
    "a bautiful mind": "a beautiful mind",
    "beautiful mind": "a beautiful mind",
    "american hisotry x": "american history x",
    "million dolar baby": "million dollar baby",
    "joe black": "meet joe black",
    "bicentennian man": "bicentennial man",
    "ther interpreter": "the interpreter",
    "poltergueist": "poltergeist",
    "will hinting": "good will hunting",
    "borkeback mountain": "brokeback mountain",
    "brokeback": "brokeback mountain",
    "borkebackmountain": "brokeback mountain",
    "the live of the others": "the lives of others",
    "the life of the others": "the lives of others",
}

speaker_map = {
    "camilo garcia": "camilo garcía",
    "constantino romero+6": "constantino romero",
    "adalberto maria merli+6": "adalberto maria merli",
    "giuppi izzo": "giuppy izzo",
    "maria moscardo": "maría moscardó",
    "massimo de ambrosio": "massimo de ambrosis",
    "robert duval": "robert duvall",
    "xavier fernandez": "xavier fernández",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the EmoFilm dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """
    paths = list(input_dir.glob("*.wav"))
    if resample:
        resample_dir = Path("resampled")
        resample_audio(paths, resample_dir)
        write_filelist(resample_dir.glob("*.wav"), "files_all")

    write_annotations({p.stem: emotion_map[p.stem[2:5]] for p in paths}, "label")

    df = pd.concat(
        pd.read_excel(input_dir / "f_m_corpus_it_es_en_legend.xlsx", sheet_name=None)
    ).set_index("file")
    df["gender"] = df.index.map(lambda x: x[0])
    df["language"] = df.index.map(lambda x: x[-2:])
    df["film"] = df["film"].str.lower().replace(film_map)
    df["speaker"] = (
        df["speaker"]
        .str.strip()
        .str.lower()
        .replace(r"^\?+$", pd.NA, regex=True)
        .str.replace("\u00a0", " ", regex=False)
        .replace(speaker_map)
    )
    for col in df.columns:
        write_annotations(df[col].to_dict(), col)


if __name__ == "__main__":
    main()
