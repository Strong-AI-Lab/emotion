"""Process the raw ASED dataset.

This assumes the file structure from the original dataset:
/.../
    01Neutral/
        *.wav
    02Fearful/
        *.wav
"""

import shutil
from pathlib import Path

import click
from joblib import delayed

from ertk.dataset import write_annotations, write_filelist
from ertk.utils import TqdmParallel

emotion_map = {
    "a": "anger",
    "h": "happiness",
    "n": "neutral",
    "f": "fear",
    "s": "sadness",
}

# These come from the Microsoft Speech API with their Amharic model.
# I am waiting for a response from the ASED authors for the actual
# sentences.
sentence_map = {
    "n01": "እህቴ እኮ በአውሮፕላን የመጣች ነው",
    "n02": "ሰላም በየቀኑ ልብስ ማጠብ ትወዳለች",
    "n03": "ነገ ስብሰባ አለ",
    "n04": "ሰኔ 30 ትምህርት ይዘጋል",
    "n05": "በቅርቡ ይመጣሉ",
    "n06": "ደህና እደሩ ነገም እንገናኝ",
    "n07": "ቡና ይጠጣል ጠብቀኝ",
    "f01": "እኔ እኮ በአውሮፕላን የመጣች ነው",
    "f02": "ሰላም በየቀኑ ልብስ ማጠብ ትወዳለች",
    "f03": "አንቺ አለቆች መምጣት ሁሉ ግቢ",
    "f04": "አረብ በሩ ተንኳኳ ልየዋ ነው መሰለኝ",
    "f05": "አንተ ቀስ ብለህ አውራ ይሰማዋል",
    "f06": "እሹ ተጋጨ የማያስከድኑ",
    "f07": "የዘመኑ ቤት ተቃጠለ",
    "h01": "እህቴ እኮ በአውሮፕላን የመጣች ነው",
    "h02": "ሰላም በየቀኑ ልብስ ማጠብ ትወዳለች",
    "h03": "አባቴ ኮ ስልክ ገዛ",
    "h04": "ዛሬኮ ልደት",
    "h05": "የብሔር በጣም ደስ ይላል",
    "h06": "አስቴር ወንድ ልጅ ወለደች",
    "h07": "ታሪኩ አዲስ ዓመት ነው",
    "s01": "እህቴ እኮ በአውሮፕላን የመጣች ነው",
    "s02": "ሰላም በየቀኑ ልብስ ማጠብ ትወዳለች",
    "s03": "የአክስት ልጅ በመኪና አደጋ ሞተ ምንጭ",
    "s04": "አጎቴ እኮ ታሞ ሐኪም ቤት ተገኘ",
    "s05": "ወያኔ ሰው ሁሉ በጦርነት አለቀ",
    "s06": "ዓይን መኪና ይዞ ተገለበጠ",
    "s07": "ወያኔ እኮ ትዝ አረፈች",
    "a01": "እህቴ እኮ በአውሮፕላን የመጣች ነው",
    "a02": "ሰላም በየቀኑ ልብስ ማጠብ ትወዳለች",
    "a03": "የራሱ ጉዳይ ለክልል",
    "a04": "በሰው ጉዳይ ጣልቃ አትግቡ",
    "a05": "እዚህ ምን ታደርጋለህ ሥራና አሰራር",
    "a06": "ከዚህ በፊት ቦታ",
    "a07": "ጦር በእጥፍ አለኝ ከአሁን በኋላ እንጃ ላይ",
}


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Process the ASED dataset at location INPUT_DIR and resample audio
    to 16 kHz 16-bit WAV audio.
    """

    paths = list(input_dir.glob("**/*.wav"))
    resample_dir = Path("resampled")
    if resample:
        resample_dir.mkdir(exist_ok=True, parents=True)
        # S4-04-02-01-29 has an upper-case S instead of lower case
        TqdmParallel(len(paths), "Copying files")(
            delayed(shutil.copyfile)(path, resample_dir / (path.stem.lower() + ".wav"))
            for path in paths
        )
    paths = list(resample_dir.glob("*.wav"))
    write_filelist(paths, "files_all")

    transcripts = {}
    for path in paths:
        sentence = path.stem[3:5]
        emotion = path.stem[0].lower()
        transcripts[path.stem] = sentence_map[f"{emotion}{sentence}"]
    write_annotations(transcripts, "transcript")

    write_annotations({p.stem: emotion_map[p.stem[0].lower()] for p in paths}, "label")
    write_annotations(
        {p.stem: ["F", "M"][int(p.stem[9:11]) - 1] for p in paths}, "gender"
    )
    write_annotations({p.stem: p.stem[3:5] for p in paths}, "sentence")
    write_annotations({p.stem: p.stem[6:8] for p in paths}, "repetition")
    write_annotations({p.stem: p.stem[-2:] for p in paths}, "speaker")
    write_annotations({p.stem: "am" for p in paths}, "language")


if __name__ == "__main__":
    main()
