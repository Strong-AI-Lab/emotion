"""List of speech corpora metadata."""

from typing import Callable, Dict, List, Set


class CorpusInfo:
    """Represents metadata for a generic speech corpus.

    Parameters:
    -----------
    name: str
        The corpus name.
    get_speaker: callable
        A function that takes a clip name as argument and returns the
        corresponding speaker.
    male_speakers: list of str
        List of male speakers.
    female_speakers: list of str
        List of female speakers.
    speakers: list of str
        List of all speakers. At least one of speakers, male_speakers,
        female_speakers must be present. If speakers is not present, it will be
        the union of male_speakers and female_speakers.
    """
    def __init__(self,
                 name: str,
                 get_speaker: Callable[[str], str],
                 male_speakers: List[str] = [],
                 female_speakers: List[str] = [],
                 speakers: List[str] = [],
                 speaker_groups: List[Set[str]] = []):
        self.name = name

        self.male_speakers = male_speakers
        self.female_speakers = female_speakers
        self.speakers = speakers
        if self.male_speakers and self.female_speakers:
            all_speakers = self.male_speakers + self.female_speakers
            if self.speakers and set(self.speakers) != set(all_speakers):
                raise ValueError("Given speaker list inconsistent with male "
                                 "and female speakers.")
            else:
                self.speakers = all_speakers
        if len(self.speakers) == 0:
            raise ValueError(
                "There must be at least one speaker for this corpus.")

        if speaker_groups:
            self.speaker_groups = speaker_groups
        else:
            self.speaker_groups = [{x} for x in self.speakers]
        self.get_speaker = get_speaker

    def get_speaker(self, name: str) -> str:
        raise NotImplementedError()

    def get_speaker_group(self, name: str) -> int:
        for idx, g in enumerate(self.speaker_groups):
            if name in g:
                return idx


corpora: Dict[str, CorpusInfo] = {
    'cafe': CorpusInfo(
        'CaFE',
        male_speakers=['01', '03', '05', '07', '09', '11'],
        female_speakers=['02', '04', '06', '08', '10', '12'],
        get_speaker=lambda n: n[:2]
    ),
    'crema-d': CorpusInfo(
        'CREMA-D',
        speakers=[
            '1042', '1070', '1030', '1087', '1061', '1086', '1026', '1017',
            '1039', '1082', '1032', '1015', '1062', '1012', '1046', '1010',
            '1014', '1064', '1080', '1023', '1056', '1066', '1035', '1074',
            '1068', '1027', '1043', '1065', '1076', '1060', '1019', '1011',
            '1075', '1008', '1006', '1025', '1053', '1058', '1085', '1069',
            '1024', '1084', '1033', '1054', '1090', '1013', '1038', '1072',
            '1036', '1088', '1071', '1005', '1057', '1029', '1020', '1073',
            '1050', '1007', '1031', '1003', '1002', '1079', '1040', '1047',
            '1077', '1078', '1049', '1051', '1041', '1052', '1083', '1016',
            '1034', '1009', '1055', '1048', '1018', '1091', '1045', '1022',
            '1004', '1089', '1067', '1059', '1063', '1001', '1021', '1028',
            '1044', '1037', '1081'
        ],
        get_speaker=lambda n: n[:4]
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
        ],
        get_speaker=lambda n: n[-9:-7]
    ),
    'emodb': CorpusInfo(
        'EMO-DB',
        male_speakers=['03', '10', '11', '12', '15'],
        female_speakers=['08', '09', '13', '14', '16'],
        get_speaker=lambda n: n[:2]
    ),
    'emofilm': CorpusInfo(
        'EmoFilm',
        speakers=['en', 'es', 'it'],
        get_speaker=lambda n: n[-2:]
    ),
    'enterface': CorpusInfo(
        'eNTERFACE',
        speakers=['s' + str(i) for i in range(1, 45) if i != 6],
        get_speaker=lambda n: n[:n.find('_')]
    ),
    'iemocap': CorpusInfo(
        'IEMOCAP',
        male_speakers=['01M', '02M', '03M', '04M', '05M'],
        female_speakers=['01F', '02F', '03F', '04F', '05F'],
        speaker_groups=[{'01M', '01F'}, {'02M', '02F'}, {'03M', '03F'},
                        {'04M', '04F'}, {'05M', '05F'}],
        get_speaker=lambda n: n[3:6]
    ),
    'jl': CorpusInfo(
        'JL-corpus',
        male_speakers=['male1', 'male2'],
        female_speakers=['female1', 'female2'],
        get_speaker=lambda n: n[:n.find('_')]
    ),
    'msp-improv': CorpusInfo(
        'MSP-IMPROV',
        male_speakers=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        female_speakers=['F01', 'F02', 'F03', 'F04', 'F05', 'F06'],
        speaker_groups=[{'M01', 'F01'}, {'M02', 'F02'}, {'M03', 'F03'},
                        {'M04', 'F04'}, {'M05', 'F05'}, {'M06', 'F06'}],
        get_speaker=lambda n: n[16:19]
    ),
    'portuguese': CorpusInfo(
        'Portuguese',
        speakers=['A', 'B'],
        get_speaker=lambda n: n[n.find('_') - 1]
    ),
    'ravdess': CorpusInfo(
        'RAVDESS',
        male_speakers=['{:02d}'.format(i) for i in range(1, 25, 2)],
        female_speakers=['{:02d}'.format(i) for i in range(2, 25, 2)],
        get_speaker=lambda n: n[-2:]
    ),
    'savee': CorpusInfo(
        'SAVEE',
        speakers=['DC', 'JE', 'JK', 'KL'],
        get_speaker=lambda n: n[:2]
    ),
    'semaine': CorpusInfo(
        'SEMAINE',
        speakers=['{:02d}'.format(i) for i in range(1, 25) if i not in [7, 8]],
        get_speaker=lambda n: n[:2]
    ),
    'shemo': CorpusInfo(
        'ShEMO',
        male_speakers=['M{:02d}'.format(i) for i in range(1, 57)],
        female_speakers=['F{:02d}'.format(i) for i in range(1, 32)],
        get_speaker=lambda n: n[:3]
    ),
    'smartkom': CorpusInfo(
        'SmartKom',
        speakers=[
            'AAA', 'AAB', 'AAC', 'AAD', 'AAE', 'AAF', 'AAG', 'AAH', 'AAI',
            'AAJ', 'AAK', 'AAL', 'AAM', 'AAN', 'AAO', 'AAP', 'AAQ', 'AAR',
            'AAS', 'AAT', 'AAU', 'AAV', 'AAW', 'AAX', 'AAY', 'AAZ', 'ABA',
            'ABB', 'ABC', 'ABD', 'ABE', 'ABF', 'ABG', 'ABH', 'ABI', 'ABJ',
            'ABK', 'ABL', 'ABM', 'ABN', 'ABO', 'ABP', 'ABQ', 'ABR', 'ABS',
            'AIS', 'AIT', 'AIU', 'AIV', 'AIW', 'AIX', 'AIY', 'AIZ', 'AJA',
            'AJB', 'AJC', 'AJD', 'AJE', 'AJF', 'AJG', 'AJH', 'AJI', 'AJJ',
            'AJK', 'AJL', 'AJM', 'AJN', 'AJO', 'AJP', 'AJQ', 'AJR', 'AJS',
            'AJT', 'AJU', 'AJV', 'AJW', 'AJX', 'AJY', 'AJZ', 'AKA', 'AKB',
            'AKC', 'AKD', 'AKE', 'AKF', 'AKG'
        ],
        get_speaker=lambda n: n[8:11]
    ),
    'tess': CorpusInfo(
        'TESS',
        speakers=['OAF', 'YAF'],
        get_speaker=lambda n: n[:3]
    ),
    'urdu': CorpusInfo(
        'URDU',
        speakers=[
            'SF1', 'SF2', 'SF3', 'SF4', 'SF5', 'SF6', 'SF7', 'SF8', 'SF9',
            'SF10', 'SF11', 'SM1', 'SM2', 'SM3', 'SM4', 'SM5', 'SM6', 'SM7',
            'SM17', 'SM18', 'SM19', 'SM20', 'SM21', 'SM22', 'SM23', 'SM24',
            'SM25', 'SM26', 'SM27'
        ],
        get_speaker=lambda n: n[:n.index('_')]
    ),
    'venec': CorpusInfo(
        'VENEC',
        speakers=[
            'AUS_01', 'AUS_02', 'AUS_03', 'AUS_04', 'AUS_05', 'AUS_06',
            'AUS_07', 'AUS_08', 'AUS_09', 'AUS_10', 'AUS_11', 'AUS_12',
            'AUS_13', 'AUS_14', 'AUS_15', 'AUS_16', 'AUS_17', 'AUS_18',
            'AUS_19', 'AUS_20', 'IND_01', 'IND_02', 'IND_03', 'IND_04',
            'IND_05', 'IND_06', 'IND_07', 'IND_08', 'IND_09', 'IND_10',
            'IND_11', 'IND_13', 'IND_14', 'IND_15', 'IND_16', 'IND_17',
            'IND_18', 'IND_19', 'IND_20', 'KEN_01', 'KEN_02', 'KEN_03',
            'KEN_04', 'KEN_05', 'KEN_06', 'KEN_07', 'KEN_08', 'KEN_09',
            'KEN_10', 'KEN_11', 'KEN_12', 'KEN_13', 'KEN_14', 'KEN_15',
            'KEN_16', 'KEN_17', 'KEN_19', 'KEN_20', 'SIN_01', 'SIN_02',
            'SIN_03', 'SIN_04', 'SIN_06', 'SIN_07', 'SIN_08', 'SIN_09',
            'SIN_13', 'SIN_14', 'USA_01', 'USA_03', 'USA_05', 'USA_06',
            'USA_07', 'USA_09', 'USA_10', 'USA_11', 'USA_12', 'USA_13',
            'USA_14', 'USA_15', 'USA_16', 'USA_17', 'USA_18', 'USA_19',
            'USA_21', 'USA_22'
        ],
        get_speaker=lambda n: n[5:]
    ),

    # Non-emotional speech datasets
    'accentdb': CorpusInfo(
        'accentDB',
        get_speaker=lambda n: n[:n.rfind('_')],
        speakers=[
            'australian_s01', 'australian_s02', 'bangla_s01', 'bangla_s02',
            'indian_s01', 'indian_s02', 'malayalam_s01', 'malayalam_s02',
            'malayalam_s03', 'odiya_s01', 'telugu_s01', 'telugu_s02',
            'welsh_s01'
        ]
    ),
    'esf': CorpusInfo(
        'ESF',
        get_speaker=lambda n: n[-2:],
        speakers=['JA', 'MA', 'RA', 'AN', 'LA', 'SA', 'VI'],
    ),
    'leap': CorpusInfo(
        'Leap',
        get_speaker=lambda n: n[:2],
        speakers=[
            'ab', 'ai', 'aj', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd',
            'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo',
            'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz',
            'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cl', 'cm',
            'cn', 'cp', 'cq', 'cr', 'dv', 'ev'
        ]
    ),
    'parole': CorpusInfo(
        'PAROLE',
        get_speaker=lambda n: n[7:10],
        speakers=[
            '001', '002', '003', '004', '005', '006', '007', '008', '009',
            '010', '011', '012', '013', '014', '015', '016', '017', '019',
            '020', '021', '022', '023', '024', '025', '027', '028', '029',
            '030', '031', '032', '033', '034', '035'
        ]
    )
}
