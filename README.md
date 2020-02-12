# Speech Emotion Recognition

This repository contains scripts for processing various emotional speech
datasets, and training machine learning models.

## Datasets Supported
Currently the following datasets are supported, along with their respective
directories:
- `cafe/` - [CaFE](https://zenodo.org/record/1478765)
- `demos/` - [DEMoS](https://zenodo.org/record/2544829)
- `emodb/` - [EMO-DB](http://emodb.bilderbar.info/)
- `emofilm/` - [EmoFilm](https://zenodo.org/record/1326428)
- `enterface/` - [eNTERFACE](http://www.enterface.net/results/)
- `iemocap/` - [IEMOCAP](https://sail.usc.edu/iemocap/)
- `jl/` - [JL-corpus](https://www.kaggle.com/tli725/jl-corpus)
- `msp-improv/` - [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
- `portuguese` - [Portuguese corpus](https://link.springer.com/article/10.3758/BRM.42.1.74)
- `ravdess/` - [RAVDESS](https://zenodo.org/record/1188976)
- `shemo/` - [ShEMO](https://github.com/mansourehk/ShEMO)
- `tess/` - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487/)

## Installation
It is advised to run the scripts in a Python virtual environment. One
can be created with the command
```
python -m venv venv
```
You must run
```
pip install -r requirements.txt
```
to install all the necessary dependencies.
