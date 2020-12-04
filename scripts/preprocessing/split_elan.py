import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile
from joblib import Parallel, delayed


def process(path: Path, out_dir: Path, prefix: str = ''):
    audio, sr = soundfile.read(path)
    assert sr == 16000, "Sample rate must be 16000 Hz."

    eaf_file = path.with_suffix('.eaf')
    xml = ET.parse(eaf_file)

    time_slots = {}
    for time_slot in xml.find('TIME_ORDER'):
        time_value = None
        if 'TIME_VALUE' in time_slot.attrib:
            time_value = time_slot.attrib['TIME_VALUE']
        time_slots[time_slot.attrib['TIME_SLOT_ID']] = time_value

    try:
        phrases = next(x for x in xml.iterfind('TIER')
                       if x.attrib['LINGUISTIC_TYPE_REF'] == 'phrase')
    except StopIteration:
        return
    group = []
    utts = []
    for annotation in phrases:
        annotation = annotation[0]
        ts1 = time_slots[annotation.attrib['TIME_SLOT_REF1']]
        ts2 = time_slots[annotation.attrib['TIME_SLOT_REF2']]
        text = annotation[0].text
        if text in ['pause', 'int']:
            if len(group) == 0:
                continue
            w = [x[2] for x in group]
            utts.append((group[0][0], group[-1][1], ' '.join(w)))
            group = []
        elif text is not None:
            group.append((ts1, ts2, text))
    if len(group) > 0:
        w = [x[2] for x in group]
        utts.append((group[0][0], group[-1][1], ' '.join(w)))

    for i, (start, end, w) in enumerate(utts):
        start = int(start)
        end = int(end)
        out_name = '{}{}_{:03d}'.format(prefix, path.stem, i)
        s_sam = int(start * sr / 1000)
        e_sam = int(end * sr / 1000)
        split = audio[s_sam:e_sam]
        out_file = out_dir / (out_name + '.wav')
        soundfile.write(out_file, split, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, nargs='+',
                        help="Input director(y|ies).")
    parser.add_argument('--output', type=Path, required=True,
                        help="Output directory.")
    parser.add_argument('--prefix', type=str, default='', help="Name prefix.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.input:
        print("Processing directory {}".format(path))
        Parallel(n_jobs=-1, prefer='processes', verbose=1)(
            delayed(process)(p, args.output, args.prefix)
            for p in path.glob('**/*.wav')
        )


if __name__ == "__main__":
    main()
