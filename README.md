# Speech Emotion Recognition

This repository contains scripts for processing various emotional speech
datasets, and training machine learning models on these datasets.

## Datasets
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

Some of these datasets, (e.g. IEMOCAP), have more specific preprocessing
scripts that must be run before the general preprocessing scripts, in
order to generate the annotations, transcripts, etc.

## Installation
This project requires Python 3.7+. It is advised to run the scripts in a
Python virtual environment. One can be created with the command
```
python -m venv .venv
```
Then you can use this virtual environment:
```
. .venv/bin/activate
```
To install the `emotion_recognition` library:
```
python setup.py install
```
Or, if you want to develop continuously:
```
python setup.py develop
```

The scripts are all located in the `scripts` directory, and are not
installed. They must be run from the root of the project directory.

### Python Dependencies
This project has a number of Python dependencies, including NumPy,
TensorFlow, PyTorch, scikit-learn and Pandas. You should run
```
pip install -r requirements.txt
```
to install all the Python dependencies.

### Third-party tools
Some third-party tools are included in this repository, under the
`third_party` directory. Currently
[auDeep](https://github.com/auDeep/auDeep),
[OpenSMILE](https://www.audeering.com/opensmile/), and
[openXBOW](https://github.com/openXBOW/openXBOW) are included as
dependencies.
