![License: MIT](https://img.shields.io/github/license/Strong-AI-Lab/emotion)
![Version](https://img.shields.io/pypi/v/ertk)
![Python version](https://img.shields.io/pypi/pyversions/ertk)
![Python wheel](https://img.shields.io/pypi/wheel/ertk)

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
applications `ertk-cli`, `ertk-dataset` and `ertk-util`. Use the
`--help` option on each one to see what commands are available.

### Running experimens
`ertk-cli` is currently used to run experiments. The `exp2` subcommand
runs experiments from a config file:
```
ertk-cli exp2 /path/to/experiment.yaml override1=val1 override2=val2
```

### Viewing and processing data
`ertk-dataset` has subcommands for viewing and processing datasets. To
view info for a dataset, after running the dataset script:
```
ertk-dataset info corpus.yaml
```

To view info for individual annotations,
```
ertk-dataset annotation speaker.csv
```

See [below](#feature-extraction) for use of `ertk-dataset process` for
feature extraction.

### Utilities
`ertk-util` has miscellaneous utility functions. The most notable is
`parallel_jobs`, which runs multiple experiments in parallel on CPUs or
GPUs. Experiments are loaded into a queue and given to the next
available worker on a free CPU thread or GPU. The main thread keeps
track of failed jobs and writes them to the failed file.
```
ertk-util parallel_jobs jobs1.txt jobs2.txt --failed failed.txt --cpus $(nproc)
```

## Feature extraction
ERTK has several feature extractors and processors built in. There are
feature extractors for OpenSMILE, openXBOW, fairseq, huggingface,
speechbrain, Keras applications, Audioset models (VGGish and YAMNet),
spectrograms, kmeans clustering, resampling, phonemisation, and voice
activity detection (VAD) trimming. To list all installed preprocessors,
run `ertk-dataset process --list_processors`.

To run a processor, use `ertk-dataset process --features processor`.
For example, to extract embeddings from the original Wav2vec model:
```
ertk-dataset process \
    files.txt \
    output.nc \
    --features fairseq \
    model_type=wav2vec \
    checkpoint=/path/to/wav2vec_large.pt
```


# Experiment configs
Experiments can be configured with a YAML config file, which specifies
the dataset(s) to load, any modifcations to annotations, the features
to load, the model type and configuration.


# Datasets
There are processing scripts for many emotional speech datasets in the
`datasets` directory. See [`datsets/README.md`](datasets/README.md) for
more information about the supported datasets and the required
processing.


# Papers
Papers that we have published will have associated code in the `papers`
directory. See [`papers/README.md`](papers/README.md) for more
information about scripts for individual publications.


# Citing
If you use ERTK, please cite our paper:
```
@inproceedings{keesingAcousticFeaturesNeural2021,
    title = {Acoustic {{Features}} and {{Neural Representations}} for {{Categorical Emotion Recognition}} from {{Speech}}},
    booktitle = {Interspeech 2021},
    author = {Keesing, Aaron and Koh, Yun Sing and Witbrock, Michael},
    year = {2021},
    month = aug,
    pages = {3415--3419},
    publisher = {{ISCA}},
    doi = {10.21437/Interspeech.2021-2217},
}
```
