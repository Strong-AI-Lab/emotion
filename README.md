# Speech Emotion Recognition
This repository contains scripts for processing various emotional speech
datasets, and training machine learning models on these datasets.

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
