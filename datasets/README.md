# Emotional speech datasets
This directory contains some scripts for processing various emotional
speech corpora into the format required by the scripts in this
repository.

Currently the following datasets are supported, along with their
respective directories in the `datasets` directory:
- `cafe` - [CaFE](https://zenodo.org/record/1478765)
- `crema-d` - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- `demos` - [DEMoS](https://zenodo.org/record/2544829)
- `emodb` - [EMO-DB](http://emodb.bilderbar.info/)
- `emofilm` - [EmoFilm](https://zenodo.org/record/1326428)
- `enterface` - [eNTERFACE](http://www.enterface.net/results/)
- `iemocap` - [IEMOCAP](https://sail.usc.edu/iemocap/)
- `jl` - [JL-corpus](https://www.kaggle.com/tli725/jl-corpus)
- `msp-improv` - [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
- `portuguese` - [Portuguese corpus](https://link.springer.com/article/10.3758/BRM.42.1.74)
- `ravdess` - [RAVDESS](https://zenodo.org/record/1188976)
- `savee` - [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)
- `semaine` - [SEMAINE](https://semaine-db.eu/)
- `shemo` - [ShEMO](https://github.com/mansourehk/ShEMO)
- `smartkom` - [SmartKom](https://clarin.phonetik.uni-muenchen.de/BASRepository/index.php)
- `tess` - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487/)
- `urdu` - [URDU](https://github.com/siddiquelatif/URDU-Dataset/)
- `venec` - [VENEC](https://www.nature.com/articles/s41562-019-0533-6)

## Processing
Each dataset has a `process.py` script that can be used to process the
data into a usable format.
```
python process.py /path/to/data
```
For most datasets this just involves resampling to 16 kHz 16-bit WAV
audio using FFmpeg, but some dataset (e.g. IEMOCAP, MSP-IMPROV) have
some more specific requirements.

All scripts will output any annotation files in CSV format, as well as
print the emotion label distribution and any inter-rater agreement
metrics that can be calculated (e.g. Krippendorf's alpha, Fleiss'
kappa).

A list of file paths to the resampled audio clips will be created for
later preprocessing scripts that extract/generate features. Usually all
audio clips will be used to generate a dataset file. If you want to only
use a subset of audio clips for processing, you can create this list
yourself. For example,
```
find /path/to/audio/subset/ -name "*.wav" | sort > files.txt
```
Multiple file lists can be created if different subsets want to be
tested independently (e.g. testing IEMOCAP's scripted and improvised
sessions independently.)

Once these steps are done, you can use the feature extraction and
dataset creation scripts.

## Labels
The format of the labels file is a simple mapping from *name* to
emotional label, where *name* is the name of the audio file without file
extension, (i.e. `name.wav` -> `name`). Some datasets (e.g. SAVEE) have
duplicate names, so the speaker is prepended to make the names unique.
The first column of the CSV is the name, and the second is the emotional
label.

Multiple label files can be created, just like multiple lists of clips.
Label files act independently of clip lists, but when a given label file
is used, it must have a label for each name in the clip list (but may
contain superfluous labels).

## Schema
The JSON metadata schema is currently unused, as most datasets use
entirely different metadata formats that would be too difficult to
unify. It may be used in future.
