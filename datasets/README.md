# Emotional speech datasets
These directories contain metadata and scripts for processing emotional
speech datasets into the format required by the scripts in this
repository.

Please see also https://github.com/SuperKogito/SER-datasets for another
list of emotional speech datasets.

Currently the following datasets have processing scripts:
- [AESDD](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)
- [ASED](https://github.com/wgwangang/ASED_V1)
- [BAVED](https://www.kaggle.com/a13x10/basic-arabic-vocal-emotions-dataset)
- [CaFE](https://zenodo.org/record/1478765)
- [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [DEMoS](https://zenodo.org/record/2544829)
- [EESC](https://metashare.ut.ee/repository/browse/estonian-emotional-speech-corpus/4d42d7a8463411e2a6e4005056b40024a19021a316b54b7fb707757d43d1a889/)
- [EMO-DB](http://emodb.bilderbar.info/)
- [EmoFilm](https://zenodo.org/record/1326428)
- [EmoryNLP](https://github.com/declare-lab/MELD/)
- [EmoV-DB](https://github.com/numediart/EmoV-DB)
- [EMOVO](http://voice.fub.it/activities/corpora/emovo/index.html)
- [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data)
- [eNTERFACE](http://www.enterface.net/results/)
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [JL-corpus](https://www.kaggle.com/tli725/jl-corpus)
- [MELD](https://github.com/declare-lab/MELD/)
- [MESD](https://data.mendeley.com/datasets/cy34mh68j9/3)
- [MESS](https://zenodo.org/record/3813437)
- [MLEndSND](https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals)
- [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
- [MSP-PODCAST](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
- [Oréau](https://zenodo.org/record/4405783)
- [Portuguese](https://link.springer.com/article/10.3758/BRM.42.1.74)
- [RAVDESS](https://zenodo.org/record/1188976)
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)
- [SEMAINE](https://semaine-db.eu/)
- [ShEMO](https://github.com/mansourehk/ShEMO)
- [SmartKom](https://clarin.phonetik.uni-muenchen.de/BASRepository/index.php)
- [SUBESCO](https://zenodo.org/record/4526477)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487/)
- [URDU](https://github.com/siddiquelatif/URDU-Dataset/)
- [VENEC](https://www.nature.com/articles/s41562-019-0533-6)
- [VIVAE](https://zenodo.org/record/4066235)

## Processing
Each dataset has a `process.py` script that can be used to process the
data into a usable format.
```
python process.py /path/to/data
```
For most datasets this involves resampling to 16 kHz 16-bit WAV, and
outputting annotations. The scripts for some datasets (e.g. IEMOCAP,
MSP-IMPROV) print additional statistics about perceptual ratings.

All scripts will output annotations in CSV format, and any inter-rater
agreement metrics that can be calculated (e.g. Krippendorf's alpha,
Fleiss' kappa) from ratings data.


## File lists and subsets
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
sessions independently). Some datasets have pre-defined file-lists (e.g.
the standard 4-class subset of IEMOCAP).

## corpus.yaml
Each dataset has a `corpus.yaml` file which simply lists the
annotations, partitions and file lists, along with a description of the
dataset and a "default" subset to use if one isn't specified. It also
has a key called `features_dir` which is used in when combining multiple
datasets with the same features.

## Annotations
An annotation is a mapping from instance name to value, stored as a CSV
file with two columns. Labels are the most important annotation, and
some datasets can have multiple different labels (e.g. acted labels vs.
perceptual labels).

## Partitions
A partition is a categorical annotation which partitions instances into
groups (e.g. label, speaker).
