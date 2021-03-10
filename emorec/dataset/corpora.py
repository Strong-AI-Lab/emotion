"""List of speech corpora metadata."""

from typing import Dict, List, Set


class CorpusInfo:
    """Represents metadata for a generic speech corpus.

    Parameters:
    -----------
    name: str
        The corpus name.
    male_speakers: list
        List of male speakers.
    female_speakers: list
        List of female speakers.
    speakers_groups: list
        List of speaker groups (sets of speakers).
    """
    def __init__(self,
                 name: str,
                 male_speakers: List[str] = [],
                 female_speakers: List[str] = [],
                 speaker_groups: List[Set[str]] = []):
        self.name = name
        self.male_speakers = male_speakers
        self.female_speakers = female_speakers
        self.speaker_groups = speaker_groups


corpora: Dict[str, CorpusInfo] = {
    'cafe': CorpusInfo(
        'CaFE',
        male_speakers=['01', '03', '05', '07', '09', '11'],
        female_speakers=['02', '04', '06', '08', '10', '12']
    ),
    'crema-d': CorpusInfo(
        'CREMA-D'
    ),
    'demos': CorpusInfo(
        'DEMoS',
        male_speakers=[
            '02', '03', '04', '05', '08', '09', '10', '11', '12', '14', '15',
            '16', '18', '19', '23', '24', '25', '26', '27', '28', '30', '33',
            '34', '39', '41', '50', '51', '52', '53', '58', '59', '63', '64',
            '65', '66', '67', '68', '69'
        ],
        female_speakers=[
            '01', '17', '21', '22', '29', '31', '36', '37', '38', '40', '43',
            '45', '46', '47', '49', '54', '55', '56', '57', '60', '61'
        ]
    ),
    'emodb': CorpusInfo(
        'EMO-DB',
        male_speakers=['03', '10', '11', '12', '15'],
        female_speakers=['08', '09', '13', '14', '16']
    ),
    'emofilm': CorpusInfo(
        'EmoFilm'
    ),
    'enterface': CorpusInfo(
        'eNTERFACE'
    ),
    'iemocap': CorpusInfo(
        'IEMOCAP',
        male_speakers=['01M', '02M', '03M', '04M', '05M'],
        female_speakers=['01F', '02F', '03F', '04F', '05F'],
        speaker_groups=[{'01M', '01F'}, {'02M', '02F'}, {'03M', '03F'},
                        {'04M', '04F'}, {'05M', '05F'}]
    ),
    'jl': CorpusInfo(
        'JL-corpus',
        male_speakers=['male1', 'male2'],
        female_speakers=['female1', 'female2']
    ),
    'msp-improv': CorpusInfo(
        'MSP-IMPROV',
        male_speakers=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        female_speakers=['F01', 'F02', 'F03', 'F04', 'F05', 'F06'],
        speaker_groups=[{'M01', 'F01'}, {'M02', 'F02'}, {'M03', 'F03'},
                        {'M04', 'F04'}, {'M05', 'F05'}, {'M06', 'F06'}]
    ),
    'portuguese': CorpusInfo(
        'Portuguese'
    ),
    'ravdess': CorpusInfo(
        'RAVDESS',
        male_speakers=[f'{i:02d}' for i in range(1, 25, 2)],
        female_speakers=[f'{i:02d}' for i in range(2, 25, 2)]
    ),
    'savee': CorpusInfo(
        'SAVEE'
    ),
    'semaine': CorpusInfo(
        'SEMAINE'
    ),
    'shemo': CorpusInfo(
        'ShEMO',
        male_speakers=[f'M{i:02d}' for i in range(1, 57)],
        female_speakers=[f'F{i:02d}' for i in range(1, 32)]
    ),
    'smartkom': CorpusInfo(
        'SmartKom'
    ),
    'tess': CorpusInfo(
        'TESS'
    ),
    'urdu': CorpusInfo(
        'URDU'
    ),
    'venec': CorpusInfo(
        'VENEC'
    )
}
