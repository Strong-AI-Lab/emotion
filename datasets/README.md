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
- `venec` - [VENEC](https://www.nature.com/articles/s41562-019-0533-6)

## Processing
Before any processing can be done, all audio must be converted to s16_le
encoded WAV, with a sample rate of 16 kHz. This can be accomplished with
the FFmpeg program:
```
ffmpeg -i file.wav -ar 16000 wav_corpus/out.wav
```
In some cases (e.g. EMO-DB) this isn't necessary as the audio is already
in this format. If you have GNU Parallel installed, you can use that to
batch process many audio files in parallel, e.g.
```
find -name "*.mp3" | parallel ffmpeg -i {} -ar 16000 wav_corpus/{/.}.wav
```

Some datasets require more customised scripts to process the data from
annotation files. Currently there are scripts for CREMA-D, IEMOCAP,
MSP-IMPROV, SEMAINE (unused for classification) and SmartKom. Each of
the scripts can be run like so, from the relevant directory:
```
python process.py
```
They will output any annotation files in CSV format, as well as print
the emotion label distribution and any inter-rater agreement metrics
that can be calculated (e.g. Krippendorf's alpha, Fleiss' kappa).

A list of audio clips must then be created, to be used for later
preprocessing scripts that extract/generate features. For example,
```
find wav_corpus -name "*.mp3" | sort > files.txt
```
Usually all audio clips will be used to generate a dataset file.
Multiple file lists can be created if different subsets want to be
tested independently (e.g. testing IEMOCAP's scripted and improvised
sessions independently.)

Once these steps are done, you can use the feature extraction and
dataset creation scripts.


## Schema
The JSON metadata schema is currently unused, as most datasets use
entirely different metadata formats that would be too difficult to
unify. It may be used in future.
