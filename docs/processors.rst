Instance processing
===================

A number of built-in processors are available to process the data,
depending on the installed packages. Additionally, custom processors
can be created through a plugin interface.

The full list of available processors can be found in the `preprocessor
documentation <api/preprocessing>`_.


Examples
--------
Suppose we want to extract eGeMAPS features for a dataset such as
EMO-DB. We can use the `ertk-cli process` command to process the
instances in EMO-DB and extract eGeMAPS features using OpenSMILE::

    ertk-cli process \
        --processor opensmile \
        datasets/EMO-DB/files_all.txt \
        datasets/EMO-DB/features/eGeMAPS.nc \
        opensmile_config=eGeMAPS

We can run an automatic speech recogniser from HuggingFace using the
``hugginface`` processor::

    ertk-cli process \
        --processor huggingface \
        datasets/EMO-DB/files_all.txt \
        datasets/EMO-DB/transcripts.txt \
        model=facebook/wav2vec2-base-960h
