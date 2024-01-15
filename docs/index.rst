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

.. [1] A. Keesing, Y. S. Koh, V. Yogarajan, and M. Witbrock, "Emotion
       Recognition ToolKit (ERTK): Standardising Tools For Emotion Recognition
       Research," in Proceedings of the 31st ACM International Conference on
       Multimedia, in MM '23. New York, NY, USA: Association for Computing
       Machinery, Oct. 2023, pp. 9693-9696. doi: 10.1145/3581783.3613459.



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
