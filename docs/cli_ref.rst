Command-line programs reference
===============================

A number of command-line tools are installed along with the ERTK Python
library. Three executables are installed:

* `ertk-cli`_
* `ertk-dataset`_
* `ertk-util`_

These each have a number of subcommands available, shown with the
``--help`` option.

ertk-cli
--------

Experiments
^^^^^^^^^^^
.. click:: ertk.cli.cli.exp:main
    :prog: ertk-cli exp

.. click:: ertk.cli.cli.exp2:main
    :prog: ertk-cli exp2

Model training and inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. click:: ertk.cli.cli.classify:main
    :prog: ertk-cli classify

.. click:: ertk.cli.cli.train:main
    :prog: ertk-cli train


ertk-dataset
------------

Info
^^^^
.. click:: ertk.cli.dataset.annotation_stats:main
    :prog: ertk-dataset annotation

.. click:: ertk.cli.dataset.info:main
    :prog: ertk-dataset info


Running processors and feature extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. click:: ertk.cli.dataset.process:main
    :prog: ertk-dataset process


Features
^^^^^^^^
.. click:: ertk.cli.dataset.combine:main
    :prog: ertk-dataset combine

.. click:: ertk.cli.dataset.convert:main
    :prog: ertk-dataset convert

.. click:: ertk.cli.dataset.remove_instances:main
    :prog: ertk-dataset remove_instances

.. click:: ertk.cli.dataset.vis:main
    :prog: ertk-dataset vis


ertk-util
---------

Run parallel CPU or GPU jobs with a simple command:

.. click:: ertk.cli.util.parallel_jobs:main
    :prog: ertk-util parallel_jobs

Other utilities
^^^^^^^^^^^^^^^
.. click:: ertk.cli.util.create_cv_dirs:main
    :prog: ertk-util create_cv_dirs

.. click:: ertk.cli.util.pgrid_to_confs:main
    :prog: ertk-util grid_to_conf

.. click:: ertk.cli.util.names_to_filenames:main
    :prog: ertk-util names_to_filenames

CHAT and ELAN file formats
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. click:: ertk.cli.util.split_chat:main
    :prog: ertk-util split_chat

.. click:: ertk.cli.util.split_elan:main
    :prog: ertk-util split_elan
