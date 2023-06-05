# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path

import sphinx

import ertk

_docs_dir = Path(__file__).parent
_src_dir = _docs_dir.parent / "src"
sys.path.insert(0, str(_src_dir.resolve()))


# -- Project information -----------------------------------------------------

project = "ERTK"
copyright = "2019, Aaron Keesing"
author = "Aaron Keesing"
version = ertk.__version__


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_rtd_theme",
    "sphinx_click",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
html_static_path = ["_static"]


primary_domain = "py"
default_role = "py:obj"

add_module_names = True
autodoc_default_options = {
    "member-order": "alphabetical",
}

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

numpydoc_show_class_members = False

add_module_names = True
