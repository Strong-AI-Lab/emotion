[![License: MIT](https://img.shields.io/github/license/Strong-AI-Lab/emotion)](LICENSE)
[![Version](https://img.shields.io/pypi/v/ertk)](https://pypi.org/project/ertk/)
[![Python version](https://img.shields.io/pypi/pyversions/ertk)](https://pypi.org/project/ertk/)
[![Python wheel](https://img.shields.io/pypi/wheel/ertk)](https://pypi.org/project/ertk/)
[![Read the Docs](https://img.shields.io/readthedocs/ertk)](https://ertk.readthedocs.io/en/stable/)
[![GitHub stars](https://img.shields.io/github/stars/Strong-AI-Lab/emotion?style=social)](https://github.com/Strong-AI-Lab/emotion)

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

### Optional dependencies
Optional dependencies can be install via:
```
pip install -r requirements-dev.txt
```
Or via PyPI:
```
pip install ertk[all-preprocessors]
```
Note that if installing from PyPI, fairseq is not updated on PyPI and so
must be installed from GitHub directly:
```
pip install git+https://github.com/facebookresearch/fairseq.git@ae59bd6d04871f6174351ad46c90992e1dca7ac7
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


# Examples
See the `examples` directory for examples.


# Citing
Our paper has now been publised at [ACM Multimedia 2023](
https://dl.acm.org/doi/10.1145/3581783.3613459). If you use ERTK, please
cite our paper:
```
@inproceedings{keesingEmotionRecognitionToolKit2023,
  title = {Emotion {{Recognition ToolKit}} ({{ERTK}}): {{Standardising Tools For Emotion Recognition Research}}},
  shorttitle = {Emotion {{Recognition ToolKit}} ({{ERTK}})},
  booktitle = {Proceedings of the 31st {{ACM International Conference}} on {{Multimedia}}},
  author = {Keesing, Aaron and Koh, Yun Sing and Yogarajan, Vithya and Witbrock, Michael},
  year = {2023},
  month = oct,
  series = {{{MM}} '23},
  pages = {9693--9696},
  publisher = {{Association for Computing Machinery}},
  address = {{New York, NY, USA}},
  doi = {10.1145/3581783.3613459},
}
```
