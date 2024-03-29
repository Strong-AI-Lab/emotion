"""
Scikit-learn backend
====================

This module contains the scikit-learn backend for ERTK. It is used to
provide a consistent interface for scikit-learn models, and to provide
additional functionality.

.. autosummary::
    :toctree: generated/

    models
    classification
    utils
"""

from .models import get_sk_model, get_sk_model_fn

__all__ = ["get_sk_model", "get_sk_model_fn"]
