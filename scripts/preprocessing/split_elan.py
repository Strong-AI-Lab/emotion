import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import click
import librosa
import soundfile
from joblib import delayed

from ertk.utils import PathlibPath, TqdmParallel


@click.command()
@click.argument("input", type=PathlibPath(exists=True), nargs=-1)
@click.argument("output", type=Path)
@click.option("--prefix", type=str, default="")
def main(input: Tuple[Path], output: Path, prefix: str):
    """Splits ELAN (.eaf) files and associated audio into segments.

    This is designed around the Leap corpus and may not work elsewhere.
    """

    def process(path: Path):
        audio, _ = librosa.load(path, sr=16000, res_type="kaiser_fast")

        eaf_file = path.with_suffix(".eaf")
        xml = ET.parse(eaf_file)

        time_order = xml.find("TIME_ORDER")
        if time_order is None:
            # No timestamps
            return

        time_slots = {}
        for time_slot in time_order:
            time_value = time_slot.attrib.get("TIME_VALUE", "")
            time_slots[time_slot.attrib["TIME_SLOT_ID"]] = time_value

        phrases = next(
            (
                x
                for x in xml.iterfind("TIER")
                if x.attrib["LINGUISTIC_TYPE_REF"] == "phrase"
            ),
            None,
        )
        if phrases is None:
            # No linguistic content
            return

        group: List[Tuple[str, str, str]] = []
        utts = []
        for annotation in phrases:
            alignable_annotation = annotation[0]
            ts1 = time_slots[alignable_annotation.attrib["TIME_SLOT_REF1"]]
            ts2 = time_slots[alignable_annotation.attrib["TIME_SLOT_REF2"]]
            annotation_value = alignable_annotation[0].text
            if annotation_value in ["pause", "int"] and len(group) > 0:
                utts.append(
                    (group[0][0], group[-1][1], " ".join([x[2] for x in group]))
                )
                group = []
            elif annotation_value is not None:
                group.append((ts1, ts2, annotation_value))
        if len(group) > 0:
            utts.append((group[0][0], group[-1][1], " ".join([x[2] for x in group])))

        for i, (start, end, w) in enumerate(utts):
            out_name = f"{prefix}{path.stem}_{i:03d}"
            print(out_name, w)
            s_sam = int(start) * 16
            e_sam = int(end) * 16
            split = audio[s_sam:e_sam]
            out_file = output / (out_name + ".wav")
            soundfile.write(out_file, split, 16000)

    output.mkdir(parents=True, exist_ok=True)
    all_files = [p for path in input for p in path.glob("**/*.wav")]
    TqdmParallel(total=len(all_files), desc="Processing ELAN files", n_jobs=-1)(
        delayed(process)(p) for p in all_files
    )


if __name__ == "__main__":
    main()
