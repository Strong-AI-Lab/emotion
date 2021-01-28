import re
from pathlib import Path
from xml.etree import ElementTree as ET

import click
import numpy as np
import pandas as pd
import soundfile


@click.command()
@click.argument('sess_dir', type=Path, default='Sessions')
@click.argument('output', type=Path, default='combined')
def main(sess_dir: Path, output: Path):
    """Processes all sessions in the SEMAINE corpus from the SESS_DIR
    directory and puts combined audio/transcripts/ratings into the
    OUTPUT directory.

    By default SESS_DIR is assumed to be `Sessions/`.
    """
    recordings = {}
    session_dirs = sess_dir.glob('*')
    for session_dir in session_dirs:
        session_id = int(session_dir.stem)

        session_file = session_dir / 'session.xml'
        session_xml = ET.parse(session_file).getroot()
        recording = int(session_xml.attrib['recording'])
        character = session_xml.attrib['character']

        files = list(session_dir.glob('*'))
        words_user_path = next((f for f in files if re.search(
            r'wordLevel_alignedTranscript.*_user', f.name)), None)
        words_operator_path = next((f for f in files if re.search(
            r'wordLevel_alignedTranscript.*_operator', f.name)), None)
        transcript_path = next((f for f in files if re.search(
            r'^alignedTranscript_[0-9]+.*\.txt', f.name)), None)
        feeltraces = [f for f in files if re.search(
            r'[AR][0-9].*\.txt', f.name)]
        annotations = {}
        for trace in feeltraces:
            match = re.search(
                r'[AR]([0-9]+)[RS]([0-9]+)TUC(Ob|Po|Pr|Sp)2?D([AEPV])\.txt',
                trace.name
            )
            if match:
                rater = int(match.group(1))
                emotion = {'A': 'Activation', 'E': 'Expectation', 'P': 'Power',
                           'V': 'Valence'}[match.group(4)]
                if emotion not in annotations:
                    annotations[emotion] = {}
                annotations[emotion][rater] = trace
        operator_audio_path = next((f for f in files if re.search(
            r'Operator HeadMounted.*\.wav', f.name)), None)
        user_audio_path = next((f for f in files if re.search(
            r'User HeadMounted.*\.wav', f.name)), None)
        if recording not in recordings:
            recordings[recording] = []
        session_info = (session_id,
                        character,
                        operator_audio_path,
                        words_operator_path,
                        user_audio_path,
                        words_user_path,
                        transcript_path,
                        annotations)
        recordings[recording].append(session_info)

    for recording, sessions in sorted(recordings.items()):
        duration = 0
        concat_user_audio = []
        concat_operator_audio = []
        concat_words_user = ''
        concat_words_operator = ''
        concat_transcript = ''
        emotion_data = {}
        sessions = sorted(sessions, key=lambda x: x[0])
        print("Recording {}:".format(recording))
        for (session_id,
             character,
             operator_audio_path,
             words_operator_path,
             user_audio_path,
             words_user_path,
             transcript_path,
             annotations) in sessions:
            if (character.lower() in ['beginning', 'end', 'forbidden']
                    or not words_user_path):
                print("\tSession {} skipped due to having no transcript."
                      .format(session_id))
                print()
                continue

            print("\tSession {}:".format(session_id))

            for emotion in annotations:
                rater_data = []
                for rater in annotations[emotion]:
                    trace = annotations[emotion][rater]
                    rater_data.append(
                        np.fromfile(trace, sep=' ').reshape((-1, 2)))

                # Truncate time to size of smallest array across raters
                smallest = np.argmin([x.shape[0] for x in rater_data])
                max_size = np.max([x.shape[0] for x in rater_data])
                num_indices = rater_data[smallest].shape[0]
                for i in range(len(rater_data)):
                    rater_data[i] = rater_data[i][:num_indices, :]

                min_size = num_indices
                print('\t', emotion, len(rater_data), rater_data[0].shape)
                if max_size - min_size > 2:
                    print("\t\t WARNING: Rating length difference: ",
                          max_size - min_size)
                mean_values = np.stack(
                    [x[:, 1] for x in rater_data], axis=0).mean(0)
                mean_raters = np.stack(
                    [rater_data[0][:, 0], mean_values], axis=1)
                mean_raters[:, 0] += (duration / 1000)

                if emotion not in emotion_data:
                    emotion_data[emotion] = mean_raters
                else:
                    emotion_data[emotion] = np.concatenate(
                        [emotion_data[emotion], mean_raters], axis=0)

            # Truncate time to size of smallest array across emotions
            smallest_emotion = min(emotion_data,
                                   key=lambda e: emotion_data[e].shape[0])
            max_size = max([emotion_data[e].shape[0] for e in emotion_data])
            num_indices = emotion_data[smallest_emotion].shape[0]
            for e in emotion_data:
                emotion_data[e] = emotion_data[e][:num_indices, :]

            min_size = num_indices
            if max_size - min_size > 2:
                print("\t WARNING: Annotation length difference: {}".format(
                    max_size - min_size))

            if transcript_path:
                print('\t', transcript_path.name)
                with open(transcript_path) as fid:
                    concat_transcript += fid.read()

            print('\t', words_user_path.name)
            with open(words_user_path) as fid:
                for line in fid:
                    m = re.search(
                        r'([0-9]+) ([0-9]+) ([A-Z\'"?.<>]+)', line)
                    if m:
                        line = '{} {} {}\n'.format(
                            int(m.group(1)) + duration,
                            int(m.group(2)) + duration,
                            m.group(3))
                    concat_words_user += line

            if words_operator_path:
                print('\t', words_operator_path.name)
                with open(words_operator_path) as fid:
                    for line in fid:
                        m = re.search(
                            r'([0-9]+) ([0-9]+) ([A-Z\'"?.<>]+)', line)
                        if m:
                            line = '{} {} {}\n'.format(
                                int(m.group(1)) + duration,
                                int(m.group(2)) + duration,
                                m.group(3))
                        concat_words_operator += line

            audio, _ = soundfile.read(user_audio_path)
            concat_user_audio.append(audio)
            audio, _ = soundfile.read(operator_audio_path)
            concat_operator_audio.append(audio)
            duration += int(1000 * len(audio) / 16000)
            print()

        if len(emotion_data) < 4:
            continue

        times = emotion_data['Valence'][:, 0]
        emotion_data = {e: x[:, 1] for e, x in emotion_data.items()}

        concat_user_audio = np.concatenate(concat_user_audio)
        concat_operator_audio = np.concatenate(concat_operator_audio)

        output_dir = output / str(recording)
        output_dir.mkdir(exist_ok=True, parents=True)
        soundfile.write(output_dir / 'user.wav', concat_user_audio,
                        samplerate=16000)
        soundfile.write(output_dir / 'operator.wav', concat_operator_audio,
                        samplerate=16000)
        with open(output_dir / 'transcript.txt', 'w') as fid:
            fid.write(concat_transcript)
        with open(output_dir / 'user.txt', 'w') as fid:
            fid.write(concat_words_user)
        with open(output_dir / 'operator.txt', 'w') as fid:
            fid.write(concat_words_operator)

        df = pd.DataFrame(emotion_data, index=pd.Index(times, name='Time'))
        df.to_csv(output_dir / 'emotions.csv')


if __name__ == "__main__":
    main()
