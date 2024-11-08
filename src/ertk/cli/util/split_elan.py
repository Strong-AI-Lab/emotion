import xml.etree.ElementTree as ET
from pathlib import Path

import click
import librosa
import soundfile
from joblib import delayed

from ertk.utils import TqdmParallel


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument("output", type=Path)
@click.option("--prefix", type=str, default="")
@click.option(
    "--tier", default="phrase", help="Tier from which to extract transcripts."
)
@click.option(
    "--break_on",
    multiple=True,
    default=["pause", "int"],
    help="Break the utterance at these annotation values.",
)
def main(
    input: tuple[Path], output: Path, prefix: str, tier: str, break_on: tuple[str]
):
    """Splits ELAN (.eaf) files and associated audio into segments."""

    def process(path: Path):
        audio, _ = librosa.load(path, sr=16000, res_type="kaiser_fast")

        eaf_file = path.with_suffix(".eaf")
        xml = ET.parse(eaf_file)

        header = xml.find("HEADER")
        if (
            header
            and "TIME_UNITS" in header.attrib
            and header.attrib["TIME_UNITS"] != "milliseconds"
        ):
            raise ValueError("`TIME_UNITS` should be 'milliseconds'.")

        time_order = xml.find("TIME_ORDER")
        if time_order is None:
            raise RuntimeError("EAF document has no `TIME_ORDER` element.")

        time_slots = {}
        for time_slot in time_order:
            time_value = time_slot.attrib.get("TIME_VALUE", "")
            time_slots[time_slot.attrib["TIME_SLOT_ID"]] = time_value

        phrases = next(
            (x for x in xml.iterfind("TIER") if x.attrib["TIER_ID"] == tier),
            None,
        )
        if phrases is None:
            # No linguistic content
            return

        group: list[tuple[str, str, str]] = []  # [(start, end, word), ...]
        utts: list[tuple[str, str, str]] = []  # [(start, end, txt), ...]
        for annotation in phrases:
            alignable_annotation = annotation[0]
            ts1 = time_slots[alignable_annotation.attrib["TIME_SLOT_REF1"]]
            ts2 = time_slots[alignable_annotation.attrib["TIME_SLOT_REF2"]]
            annotation_value = alignable_annotation[0].text
            if annotation_value in break_on and len(group) > 0:
                utts.append(
                    (group[0][0], group[-1][1], " ".join([x[2] for x in group]))
                )
                group = []
            elif annotation_value is not None:
                group.append((ts1, ts2, annotation_value))
        if len(group) > 0:
            utts.append((group[0][0], group[-1][1], " ".join([x[2] for x in group])))

        out_txt = output / f"{prefix}{path.stem}.txt"
        with open(out_txt, "w") as fid:
            for i, (start, end, w) in enumerate(utts):
                out_name = f"{prefix}{path.stem}_{i:03d}"
                print(out_name, w)
                fid.write(f"{out_name}\t{w}\n")
                s_sam = int(start) * 16
                e_sam = int(end) * 16
                out_audio = output / f"{out_name}.wav"
                soundfile.write(out_audio, audio[s_sam:e_sam], 16000)

    output.mkdir(parents=True, exist_ok=True)
    all_files = [p for path in input for p in path.glob("**/*.wav")]
    TqdmParallel(total=len(all_files), desc="Processing ELAN files", n_jobs=-1)(
        delayed(process)(p) for p in all_files
    )


if __name__ == "__main__":
    main()
