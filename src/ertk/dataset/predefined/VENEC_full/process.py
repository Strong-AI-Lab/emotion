"""Process the raw full VENEC dataset.

This assumes the file structure from the original compressed file(s):
/.../
    *.wav
"""

import re
from pathlib import Path

import click

from ertk.dataset import resample_audio, write_annotations, write_filelist

RE_EMO = re.compile(r"^(([A-Za-z]+)\d+)([A-Za-z]+)[0-9]?([LMHW][a-z]*)$")
RE_NEU = re.compile(r"^(([A-Za-z]+)\d+)([A-Z][a-z]+|Neu.*)[0-9]?$")
EMO_MAP = {
    "ang": "anger",
    "con": "contempt",
    "fea": "fear",
    "fearl": "fear",
    "hap": "happiness",
    "int": "interest",
    "lus": "lust",
    "neu": "neutral",
    "pri": "pride",
    "rel": "relief",
    "sad": "sadness",
    "sha": "shame",
    "sham": "shame",
    # full set
    "aff": "affection",
    "amu": "amusement",
    "disg": "disgust",
    "disgustr": "disgust",
    "dist": "distress",
    "gui": "guilt",
    "guil": "guilt",
    "nsur": "negativesurprise",
    "negs": "negativesurprise",
    "negsurprise": "negativesurprise",
    "nsu": "negativesurprise",
    "psur": "positivesurprise",
    "poss": "positivesurprise",
    "possurprise": "positivesurprise",
    "ser": "serenity",
}
SUBSET_11CLASS = [
    "anger",
    "contempt",
    "fear",
    "happiness",
    "interest",
    "lust",
    "neutral",
    "pride",
    "relief",
    "sadness",
    "shame",
]

INTENSITY_MAP = {
    "h": "high",
    "hi": "high",
    "lo": "low",
    "med": "medium",
    "modeare": "medium",
    "moderate": "medium",
    "weak": "low",
}

COUNTRY_MAP = {"AUS": "au", "IND": "in", "SIN": "sg", "KEN": "ke", "USA": "us"}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the VENEC dataset at location INPUT_DIR and resample
    audio to 16 kHz 16-bit WAV audio.
    """

    paths = [x for x in input_dir.glob("*") if x.suffix.lower() == ".wav"]
    resample_dir = Path("resampled")
    if resample:
        resample_audio(paths, resample_dir)
    files_all = list(resample_dir.glob("*.wav"))
    write_filelist(files_all, "files_all")

    emo_dict = {}
    speaker_dict = {}
    intensity_dict = {}
    country_dict = {}
    for p in paths:
        for reg in [RE_EMO, RE_NEU]:
            m = reg.match(p.stem)
            if m:
                speaker_dict[p.stem] = m[1].lower()
                country_dict[p.stem] = m[2].upper()  # consistent with other VENEC
                emo_dict[p.stem] = m[3].lower()
                if len(m.groups()) == 4:
                    intensity_dict[p.stem] = m[4].lower()
                else:
                    intensity_dict[p.stem] = "neutral"

    # Exceptions
    intensity_dict["Ken13Inthigh"] = "high"
    emo_dict["Ken13Inthigh"] = "int"
    intensity_dict["Sin15Angmed"] = "med"
    emo_dict["Sin15Angmed"] = "ang"
    intensity_dict["Ind4Contemptweak"] = "weak"
    emo_dict["Ind4Contemptweak"] = "contempt"
    intensity_dict["Aus011Contemptlow"] = "low"
    emo_dict["Aus011Contemptlow"] = "contempt"
    intensity_dict["Am17GuiMEd"] = "med"
    emo_dict["Am17GuiMEd"] = "gui"
    speaker_dict["Am17GuiMEd"] = "am17"
    country_dict["Am17GuiMEd"] = "USA"
    intensity_dict["Ken4SerVoc"] = "high"
    emo_dict["Ken4SerVoc"] = "ser"
    speaker_dict["Ken4SerVoc"] = "ken4"
    country_dict["Ken4SerVoc"] = "KEN"
    emo_dict["Am11NeuP2"] = "neu"

    country_dict = {k: "USA" if v == "AM" else v for k, v in country_dict.items()}
    country_dict = {k: COUNTRY_MAP[v] for k, v in country_dict.items()}
    intensity_dict = {k: INTENSITY_MAP.get(v, v) for k, v in intensity_dict.items()}
    emo_dict = {k: EMO_MAP.get(v, v) for k, v in emo_dict.items()}
    write_annotations(emo_dict, "label")
    write_annotations(speaker_dict, "speaker")
    write_annotations(country_dict, "country")
    write_annotations(intensity_dict, "intensity")
    write_annotations({p.stem: "en" for p in paths}, "language")
    write_filelist(
        {x for x in files_all if emo_dict[x.stem] in SUBSET_11CLASS}, "files_11class"
    )


if __name__ == "__main__":
    main()
