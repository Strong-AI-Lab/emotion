![License: MIT](https://img.shields.io/github/license/Strong-AI-Lab/emotion)

This repository contains scripts for processing emotional speech
datasets, and training machine learning models on the datasets. The code
was developed mainly to facilitate my PhD research, but I have tried to
make the Python library more generally useful and full of utilities for
processing datasets and training/testing models.

# Papers
See [`papers/README.md`](papers/README.md) for more information about
scripts for individual publications.

# Datasets
See [`datsets/README.md`](datasets/README.md) for more information about
the supported datasets and the required processing.

# ERTK: Emotion Recognition ToolKit

This is a Python library with utilities for processing (emotional)
speech datasets and training/testing models. There are also associated
scripts for individual tasks.

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

### Dependencies
This project has a number of Python dependencies, including NumPy,
TensorFlow, PyTorch, scikit-learn and Pandas. You should run
```
pip install -r requirements.txt
```
to install all the Python dependencies.

### Install ERTK
To install the `ertk` library:
```
python setup.py install
```
Or, if you want to develop continuously:
```
python setup.py develop
```

Note that this will not install the [scripts](#Scripts).

### Third-party tools
Some third-party tools are included in this repository, under the
`third_party` directory. See the [README](third_party/README.md) for
more details.

## Scripts
Scripts are contained in the [`scripts/`](scripts/) directory, in a
subdirectory roughly corresponding to their function. Each script should
have usage info if you give `--help` as an argument.
