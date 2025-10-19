import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import click
import librosa
import numpy as np
import pandas as pd
import soundfile


@dataclass
class SessionInfo:
    session_id: int
    character: str
    operator_audio_path: Path | None = None
    words_operator_path: Path | None = None
    user_audio_path: Path | None = None
    words_user_path: Path | None = None
    transcript_path: Path | None = None
    annotations: dict[str, dict[int, Path]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--resample/--noresample", default=True)
def main(input_dir: Path, resample: bool):
    """Processes all sessions in the SEMAINE corpus and puts combined
    audio/transcripts/ratings into the OUTPUT directory.
    """
    recordings: dict[str, list[SessionInfo]] = defaultdict(list)
    for session_dir in (input_dir / "Sessions").glob("*"):
        session_file = session_dir / "session.xml"
        session_xml = ET.parse(session_file).getroot()

        feeltraces = []
        session_info = SessionInfo(
            int(session_dir.stem), session_xml.attrib["character"]
        )
        for path in session_dir.glob("*"):
            if re.search(r"wordLevel_alignedTranscript.*_user", path.name):
                session_info.words_user_path = path
            elif re.search(r"wordLevel_alignedTranscript.*_operator", path.name):
                session_info.words_operator_path = path
            elif re.search(r"^alignedTranscript_[0-9]+.*\.txt", path.name):
                session_info.transcript_path = path
            elif re.search(r"Operator HeadMounted.*\.wav", path.name):
                session_info.operator_audio_path = path
            elif re.search(r"User HeadMounted.*\.wav", path.name):
                session_info.user_audio_path = path
            elif re.search(r"[AR][0-9].*\.txt", path.name):
                feeltraces.append(path)
        for trace in feeltraces:
            match = re.search(
                r"[AR]([0-9]+)[RS]([0-9]+)TUC(Ob|Po|Pr|Sp)2?D([AEPV])\.txt", trace.name
            )
            if match:
                rater = int(match.group(1))
                emotion = {
                    "A": "Activation",
                    "E": "Expectation",
                    "P": "Power",
                    "V": "Valence",
                }[match.group(4)]
                session_info.annotations[emotion][rater] = trace
        recordings[session_xml.attrib["recording"]].append(session_info)

    df = pd.DataFrame(columns=["Activation", "Expectation", "Power", "Valence"])
    for recording in sorted(recordings, key=lambda x: int(x)):
        duration = 0
        concat_user_audio: list[np.ndarray] = []
        concat_operator_audio: list[np.ndarray] = []
        concat_words_user = ""
        concat_words_operator = ""
        concat_transcript = ""
        emotion_data = {}
        sessions = sorted(recordings[recording], key=lambda x: x.session_id)
        print(f"Recording {recording}:")
        for session in sessions:
            duration = int(1000 * sum(len(x) for x in concat_operator_audio) / 16000)
            audio, _ = librosa.load(session.user_audio_path, sr=16000)
            concat_user_audio.append(audio)
            audio, _ = librosa.load(session.operator_audio_path, sr=16000)
            concat_operator_audio.append(audio)

            if (
                session.character.lower() in ["beginning", "end", "forbidden"]
                or not session.words_user_path
            ):
                print(
                    f"\tSession {session.session_id} ({recording}/{session.character}) "
                    "skipped"
                )
                if not session.words_user_path:
                    print("\t(Does not have user words.)")
                print(f"\t{duration}")
                print()
                continue

            print(f"\tSession {session.session_id}:")

            for emotion in session.annotations:
                rater_data = []
                for rater in session.annotations[emotion]:
                    trace = session.annotations[emotion][rater]
                    rater_data.append(np.fromfile(trace, sep=" ").reshape((-1, 2)))

                # Truncate time to size of smallest array across raters
                smallest = np.argmin([x.shape[0] for x in rater_data])
                max_size = np.max([x.shape[0] for x in rater_data])
                num_indices = rater_data[smallest].shape[0]
                for i in range(len(rater_data)):
                    rater_data[i] = rater_data[i][:num_indices, :]

                min_size = num_indices
                print("\t", emotion, len(rater_data), rater_data[0].shape)
                if (max_size - min_size) / min_size > 0.01:
                    print(
                        "\t\t WARNING: Rating length difference: ", max_size - min_size
                    )
                mean_values = np.stack([x[:, 1] for x in rater_data], axis=0).mean(0)
                mean_raters = np.stack([rater_data[0][:, 0], mean_values], axis=1)
                mean_raters[:, 0] += duration / 1000

                if emotion not in emotion_data:
                    emotion_data[emotion] = mean_raters
                else:
                    emotion_data[emotion] = np.concatenate(
                        [emotion_data[emotion], mean_raters], axis=0
                    )

            # Truncate time to size of smallest array across emotions
            smallest_emotion = min(emotion_data, key=lambda e: emotion_data[e].shape[0])
            max_size = max([emotion_data[e].shape[0] for e in emotion_data])
            num_indices = emotion_data[smallest_emotion].shape[0]
            for e in emotion_data:
                emotion_data[e] = emotion_data[e][:num_indices, :]

            min_size = num_indices
            if max_size - min_size > 2:
                print(
                    "\t WARNING: Annotation length difference: "
                    f"{max_size - min_size}"
                )

            if session.transcript_path:
                with open(session.transcript_path) as fid:
                    concat_transcript += fid.read()
            else:
                print("\tMissing transcript")

            with open(session.words_user_path) as fid:
                for line in fid:
                    m = re.search(r'([0-9]+) ([0-9]+) ([A-Z\'"?.<>]+)', line)
                    if m:
                        line = "{} {} {}\n".format(
                            int(m.group(1)) + duration,
                            int(m.group(2)) + duration,
                            m.group(3),
                        )
                    concat_words_user += line

            if session.words_operator_path:
                with open(session.words_operator_path) as fid:
                    for line in fid:
                        m = re.search(r'([0-9]+) ([0-9]+) ([A-Z\'"?.<>]+)', line)
                        if m:
                            line = "{} {} {}\n".format(
                                int(m.group(1)) + duration,
                                int(m.group(2)) + duration,
                                m.group(3),
                            )
                        concat_words_operator += line
            else:
                print("\tMissing operator words")
            print()

        if len(emotion_data) < 4:
            print(f"Recording {recording} has incomplete emotional annotations")
            print()
            continue

        times = emotion_data["Valence"][:, 0]
        emotion_data = {e: x[:, 1] for e, x in emotion_data.items()}

        concat_user_audio = np.concatenate(concat_user_audio)
        concat_operator_audio = np.concatenate(concat_operator_audio)

        output_dir = Path("combined") / recording
        output_dir.mkdir(exist_ok=True, parents=True)
        soundfile.write(output_dir / "user.wav", concat_user_audio, samplerate=16000)
        soundfile.write(
            output_dir / "operator.wav", concat_operator_audio, samplerate=16000
        )
        with open(output_dir / "transcript.txt", "w") as fid:
            fid.write(concat_transcript)
        with open(output_dir / "user.txt", "w") as fid:
            fid.write(concat_words_user)
        with open(output_dir / "operator.txt", "w") as fid:
            fid.write(concat_words_operator)

        emotions = pd.DataFrame(emotion_data, index=pd.Index(times, name="Time"))
        emotions.to_csv(output_dir / "emotions.csv")

        user_turns: dict[int, list[tuple[int, int, str]]] = {}
        operator_turns: dict[int, list[tuple[int, int, str]]] = {}
        for d, s in [
            (user_turns, concat_words_user),
            (operator_turns, concat_words_operator),
        ]:
            turn = 0
            for line in s.split("\n"):
                if line.startswith("---"):
                    m = re.search(r"---recording.*turn ([0-9]+)---", line)
                    turn = int(m.group(1))
                else:
                    m = re.search(r"([0-9]+) ([0-9]+) <?([A-Z\'?!]+)>?", line)
                    if m:
                        start = int(m.group(1)) * 16
                        end = int(m.group(2)) * 16
                        word = m.group(3)
                        if turn not in d:
                            d[turn] = []
                        d[turn].append((start, end, word))

        for d, p, audio in [
            (user_turns, "u", concat_user_audio),
            (operator_turns, "o", concat_operator_audio),
        ]:
            out_dir = output_dir / "turns"
            out_dir.mkdir(exist_ok=True)
            for turn, words in sorted(d.items()):
                start = words[0][0]
                end = words[-1][1]

                name = f"{int(recording):02d}_{p}_{turn:03d}"
                filename = name + ".wav"
                soundfile.write(out_dir / filename, audio[start:end], samplerate=16000)

                if p == "u":
                    start_idx, end_idx = np.searchsorted(
                        emotions.index, [start / 16000, end / 16000]
                    )
                    if start_idx != end_idx:
                        mean_emotions = emotions.iloc[start_idx:end_idx, :].mean()
                        df.loc[name, :] = mean_emotions

    df.index.name = "Name"
    for c in df.columns:
        df[c].to_csv(f"{c.lower()}.csv")


if __name__ == "__main__":
    main()
