from pathlib import Path

import click
from click_option_group import optgroup


def apply_decorators(f, *decorators):
    for dec in reversed(decorators):
        f = dec(f)
    return f


def dataset_args(f):
    decs = [
        click.argument(
            "corpus_info",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            nargs=-1,
        ),
        optgroup.group("Dataset options"),
        optgroup.option(
            "--subset", multiple=True, default=["default"], help="Subset selection."
        ),
        optgroup.option("--map_groups", multiple=True, help="Group name mapping."),
        optgroup.option(
            "--sel_groups",
            multiple=True,
            help="Group selection. This is a map from partition to group(s).",
        ),
        optgroup.option(
            "--remove_groups",
            multiple=True,
            help="Group deletion. This is a map from partition to group(s).",
        ),
        optgroup.option(
            "--clip_seq",
            type=int,
            help="Clip sequences to this length (before pad).",
        ),
        optgroup.option(
            "--pad_seq",
            type=int,
            help="Pad sequences to multiple of this length (after clip).",
        ),
    ]
    return apply_decorators(f, *decs)


def eval_args(f):
    decs = [
        optgroup.group("Evaluation options:"),
        optgroup.option("--cv_part", help="Partition for LOGO CV."),
        optgroup.option(
            "--kfold",
            type=int,
            help="k when using (group) k-fold cross-validation, or leave-one-out.",
        ),
        optgroup.option(
            "--inner_kfold",
            type=int,
            default=2,
            help="k for inner k-fold CV (where relevant). If -1 then LOGO is used. If "
            "1 then a random split is used.",
        ),
        optgroup.option(
            "--test_size", type=float, default=0.2, help="Test size when kfold=1."
        ),
        optgroup.option(
            "--inner_part", help="Which partition to use for group-based inner CV."
        ),
        optgroup.option("--train", help="Train data."),
        optgroup.option("--valid", help="Validation data."),
        optgroup.option("--test", help="Test data."),
        optgroup.option(
            "--inner_cv/--noinner_cv",
            "use_inner_cv",
            default=True,
            help="[deprecated] Whether to use inner CV. This is deprecated and only "
            "exists for backwards compatibility.",
        ),
    ]
    return apply_decorators(f, *decs)


def model_args(f):
    decs = [
        optgroup.group("Model options"),
        optgroup.option("--clf", "clf_type", required=True, help="Classifier to use."),
        optgroup.option(
            "--clf_args",
            "--model_args",
            "clf_args_file",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            multiple=True,
            help="File containing keyword arguments to give to model initialisation.",
        ),
        optgroup.option(
            "--param_grid",
            "param_grid_file",
            type=click.Path(exists=True, path_type=Path),
            multiple=True,
            help="File with parameter grid data.",
        ),
    ]
    return apply_decorators(f, *decs)


def train_args(f):
    decs = [
        optgroup.group("Training options"),
        optgroup.option("--features", required=True, help="Features to load."),
        optgroup.option("--label", default="label", help="Label annotation to use."),
        optgroup.option("--learning_rate", type=float, default=1e-4, show_default=True),
        optgroup.option("--batch_size", type=int, default=64, show_default=True),
        optgroup.option("--epochs", type=int, default=50, show_default=True),
        optgroup.option(
            "--balanced/--imbalanced", default=True, help="Balances sample weights."
        ),
        optgroup.option(
            "--sample_rate",
            type=int,
            help="Sample rate if loading raw audio.",
        ),
        optgroup.option(
            "--n_gpus",
            type=int,
            default=1,
            show_default=True,
            help="Number of GPUs to use.",
        ),
        optgroup.option(
            "--reps",
            type=int,
            default=1,
            show_default=True,
            help="The number of repetitions to do per test.",
        ),
        optgroup.option(
            "--normalise",
            default="online",
            show_default=True,
            help="Normalisation method. 'online' means use training data for "
            "normalisation.",
        ),
        optgroup.option(
            "--seq_transform",
            default="feature",
            show_default=True,
            help="Normalisation method for sequences.",
        ),
        optgroup.option(
            "--transform",
            type=click.Choice(["std", "minmax"]),
            default="std",
            show_default=True,
            help="Transformation class.",
        ),
        optgroup.option(
            "--n_jobs", type=int, default=-1, help="Number of parallel executions."
        ),
        optgroup.option(
            "--verbose",
            type=int,
            default=0,
            help="Verbosity. -1=nothing, 0=dataset+results, 1=INFO, 2=DEBUG",
        ),
        optgroup.option(
            "--train_config",
            "train_config_path",
            type=click.Path(exists=True, path_type=Path),
            help="Path to train config file.",
        ),
    ]
    return apply_decorators(f, *decs)


def result_args(f):
    decs = [
        optgroup.group("Results options"),
        optgroup.option("--results", type=Path, help="Results directory."),
        optgroup.option("--logdir", type=Path, help="TF/PyTorch logs directory."),
    ]
    return apply_decorators(f, *decs)
