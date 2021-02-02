![License: MIT](https://img.shields.io/github/license/Broad-Ai-Lab/emotion)

# Speech Emotion Recognition
This repository contains scripts for processing emotional speech
datasets, and training machine learning models on the datasets.

## Datasets
See [`datsets/README.md`](datasets/README.md) for more information about
the supported datasets and the required processing.

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
dependencies. The binaries of OpenSMILE are distributed along with stock
and custom config files. The openXBOW JAR file is also distributed. The
relevant LICENCE files are in each directory.

## Scripts
All scripts are contained in the [`scripts/`](scripts/) directory. The
four main types of scripts are preprocessing, training, utility and
results analysis. Each script should have usage info if you give
`--help` as an argument.

There are also some scripts for batch processing all corpora for
classification experiments, although these are subject to change until
a paper is published, in which case they will be put in the
[`papers/`](papers/) directory.

## Papers
The [`papers/`](papers/) directory contains copies of scripts used to
run experiments and results for a given paper. Each directory
corresponds to a publicaiton and has a README with a description of the
paper and how to run the experiments. Each publication will also be
associated with a git tag and a GitHub release on the
[releases](https://github.com/Broad-AI-Lab/emotion/releases) page.
