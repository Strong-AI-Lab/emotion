Running experiments
===================

Experiments can be specified with a YAML configuration file. The
configuration file specifies the dataset(s) to load, the model to use,
the training parameters, and the evaluation parameters.

The `ertk-cli exp2 <api/cli>`_ command is designed to run experiments
from a configuration file and takes optional parameter overrides on the
command line::

    ertk-cli exp2 <config_file> <overrides>

Example with EMO-DB
-------------------
Suppose we want to run a quick experiment with `EMO-DB <datasets>`_.
After downloading the dataset, standardising using the build-in script,
and extracting features, we can run an experiment.

The following is an example of such a configuration file:

.. code-block:: yaml
    :caption: exp_config.yaml

    # Configuration file for an experiment
    # ------------------------------------
    # This file specifies the dataset(s) to load, the model to use,
    # the training parameters, and the evaluation parameters.
    name: EMO-DB
    data:
        datasets:
            EMO-DB:
                path: datasets/EMO-DB/corpus.yaml
        features: eGeMAPS

    # Model parameters
    model:
        # Multinomial logistic regression
        type: sk/lr
        config:
            penalty: l2
            class_weight: null
            solver: lbfgs
            max_iter: 100
            C: 1
            multi_class: multinomial

    # Evaluation parameters
    eval:
        # Speaker-independent cross-validtion
        cv:
            part: speaker
            kfold: -1
        # Two-fold speaker-independent inner cross-validation for
        # hyperparameter tuning
        inner_kfold: 2
        inner_part: speaker

    results: results.csv

    # Training parameters
    train:
        # Run cross-validation folds in parallel
        n_jobs: -1

If we use this configuration file, we can run the experiment with the
following command::

    ertk-cli exp2 exp_config.yaml

Overrides can be specified on the command line. For example, to run
the experiment with a different cost parameter C, we can use the
following command::

    ertk-cli exp2 exp_config.yaml train.config.C=0.1
