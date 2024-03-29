"""
Preprocessor classes and plugins
================================

This module contains base classes for audio clip processors, feature
extractors and instance processors, as well as a number of plugins for
each of these.


Base classes
------------
.. autosummary::
    :toctree: generated/

    InstanceProcessor
    AudioClipProcessor
    FeatureExtractor


Plugins
-------
.. autosummary::
    :toctree: generated/

    audioset
    encodec
    fairseq
    huggingface
    keras_apps
    kmeans
    opensmile
    openxbow
    phonemize
    resample
    spectrogram
    speechbrain
    vad_trim
"""

from ._base import AudioClipProcessor, FeatureExtractor, InstanceProcessor

__all__ = ["AudioClipProcessor", "FeatureExtractor", "InstanceProcessor"]
