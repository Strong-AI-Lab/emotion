"""
fairseq extractor
==================

This module contains the extractor for the original wav2vec and
vq-wav2vec models

.. autosummary::
    :toctree:

    FairseqExtractor
    FairseqExtractorConfig
"""

from .fairseq import FairseqExtractor, FairseqExtractorConfig

__all__ = ["FairseqExtractor", "FairseqExtractorConfig"]
