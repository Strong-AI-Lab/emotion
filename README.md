![License: MIT](https://img.shields.io/github/license/Strong-AI-Lab/emotion)

# ERTK: Emotion Recognition ToolKit
This is a Python library with utilities for processing emotional
speech datasets and training/testing models. There are also command-line
tools for common tasks.

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

### Install from PyPI
You can install ERTK from [PyPI](https://pypi.org/project/ertk/) using
```
pip install ertk
```

### Install from repository
Alternatively you can clone this repository and install using the latest
commit:
```
pip install -r requirements.txt
pip install .
```
Or, if you want to develop continuously:
```
pip install -e .
```

## Using CLI tools
Upon installation, you should be able to use common tools using the CLI
applications `ertk_cli`, `ertk_dataset` and `ertk_utils`. Use the
`--help` option on each one to see what commands are available.


# Datasets
See [`datsets/README.md`](datasets/README.md) for more information about
the supported datasets and the required processing.

# Papers
Papers that we have published will have associated code in the `papers`
directory. See [`papers/README.md`](papers/README.md) for more
information about scripts for individual publications.
