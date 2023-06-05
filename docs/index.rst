ERTK documentation
==================

ERTK: Emotion Recognition ToolKit is a python library for processing
emotional speech datasets, extracting common features, and
training/testing models. It is built upon scikit-learn, with
infrastructure for PyTorch and TensorFlow models.

Citing
------
If you use ERTK in your research, please cite our paper [1]_:

.. code-block:: bibtex

    @inproceedings{keesingAcousticFeaturesNeural2021,
        title = {Acoustic Features and Neural Representations for Categorical Rmotion Recognition from Speech},
        booktitle = {Interspeech 2021},
        author = {Keesing, Aaron and Koh, Yun Sing and Witbrock, Michael},
        year = {2021},
        month = aug,
        pages = {3415--3419},
        publisher = {{ISCA}},
        doi = {10.21437/Interspeech.2021-2217}
    }

.. [1] A. Keesing, Y. S. Koh, and M. Witbrock, "Acoustic Features and Neural
       Representations for Categorical Emotion Recognition from Speech," in
       Interspeech 2021, ISCA, Aug. 2021, pp. 3415-3419. doi:
       10.21437/Interspeech.2021-2217


.. toctree::
    :maxdepth: 1
    :caption: Contents

    installation
    datasets
    processors
    experiments


.. toctree::
    :maxdepth: 1
    :caption: CLI reference

    cli_ref


.. toctree::
    :maxdepth: 1
    :caption: API reference

    api/dataset
    api/preprocessing
    api/train
    api/config
    api/cli
    api/utils


.. toctree::
    :caption: Index
    :maxdepth: 1

    genindex
