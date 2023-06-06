Installation
============

ERTK
----

ERTK can be installed from `PyPI <https://pypi.org/project/ertk/>`_
using pip::

    pip install ertk

Alternatively, you can clone the repository and install it manually::

    git clone https://github.com/Strong-AI-Lab/emotion.git
    cd emotion
    pip install .

To install in editable mode, use the ``-e`` flag::

    pip install -e .

You can also install directly from GitHub with pip::

    pip install git+https://github.com/Strong-AI-Lab/emotion.git


Optional third-party libraries
------------------------------

ERTK can use the following third-party libraries if they are installed:

* `HuggingFace Transformers
  <https://github.com/huggingface/transformers>`_ for using
  transformer-based models for feature extraction, ASR, etc.
* `fairseq <https://github.com/facebookresearch/fairseq>`_ for
  using fairseq models for feature extraction, etc.
* `phonemizer <https://github.com/bootphon/phonemizer>`_ for
  phonemizing text.
* `openSMILE <https://github.com/audeering/opensmile-python>`_ for
  feature extraction.
* `resampy <https://github.com/bmcfee/resampy>`_ for audio resampling.
* `SpeechBrain <https://github.com/speechbrain/speechbrain>`_ for
  SpeechBrain models.
* `Encodec <https://github.com/facebookresearch/encodec>`_ for feature
  extraction.
* `TensorFlow-Slim <https://github.com/google-research/tf-slim>`_ to run
  the audioset models VGGish and YAMNet.

To install ERTK with these dependencies, use the ``[all]`` extra::

    pip install ertk[all-preprocessors]

Note that fairseq is not updated on PyPI and so must be installed from
GitHub directly::

    pip install git+https://github.com/facebookresearch/fairseq.git@ae59bd6d04871f6174351ad46c90992e1dca7ac7
