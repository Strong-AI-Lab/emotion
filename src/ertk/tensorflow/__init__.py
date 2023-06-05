"""
TensorFlow backend
==================

This package contains the TensorFlow backend for the ERTK.

.. autosummary::
    :toctree: generated/

    models
    train
    dataset
    classification
    utils
"""

from .models import get_tf_model, get_tf_model_fn
from .utils import compile_wrap, init_gpu_memory_growth, test_fit
