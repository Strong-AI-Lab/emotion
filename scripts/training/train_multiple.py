import pickle
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
)

from emorec.dataset import CombinedDataset, LabelledDataset
from emorec.sklearn.models import PrecomputedSVC
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--save", type=Path, help="Location to save the model.")
@click.option(
    "--cv",
    type=click.Choice(["speaker", "corpus"]),
    default="speaker",
    help="Cross-validation method.",
)
@click.option(
    "--norm",
    type=click.Choice(["speaker", "corpus"]),
    default="speaker",
    help="Normalisation method.",
)
def main(input: Tuple[Path], save: Path, cv: str, norm: str):
    """Trains a model on the given INPUT models and labels. INPUT files
    must be pairs, where each features file is followed by a labels
    file.

    Optionally saves the model as a pickle.
    """

    if len(input) == 0:
        raise RuntimeError("No input files specified.")
    elif len(input) % 2 != 0:
        raise RuntimeError("Need a label file for each features file.")
    labels = [input[i] for i in range(1, len(input), 2)]
    features = [input[i] for i in range(0, len(input), 2)]

    dataset = CombinedDataset(
        *(
            LabelledDataset(path, lab, lab.parent / "speaker.csv")
            for path, lab in zip(features, labels)
        )
    )
    print("Loaded all datasets and labels.")
    emotion_map = {x: "emotional" for x in dataset.classes}
    emotion_map["neutral"] = "neutral"
    dataset.map_classes(emotion_map)
    dataset.remove_classes(["emotional", "neutral"])
    dataset.normalise(scheme=norm)
    print(dataset)

    splitter = LeaveOneGroupOut()
    groups = None
    if cv == "speaker":
        groups = dataset.group_indices
        if len(dataset.speaker_names) > 10:
            splitter = GroupKFold(5)
        print("Using speaker-independent cross-validation.")
    elif cv == "corpus":
        groups = dataset.corpus_indices
        print("Using corpus-independent cross-validation.")
    else:
        splitter = KFold(5)

    param_grid = {
        "C": 2.0 ** np.arange(-6, 7, 2),
        "gamma": 2.0 ** np.arange(-12, -1, 2),
    }
    clf = GridSearchCV(
        PrecomputedSVC(kernel="rbf", probability=True),
        param_grid,
        scoring="balanced_accuracy",
        cv=GroupKFold(2),
        n_jobs=2,
    )

    class_weight = dataset.n_instances / (dataset.n_classes * dataset.class_counts)
    sample_weight = class_weight[dataset.y]

    scoring = {
        "war": get_scorer("accuracy"),
        "uar": get_scorer("balanced_accuracy"),
        "recall": make_scorer(recall_score, pos_label=0),
        "precision": make_scorer(precision_score, pos_label=0),
        "f1": make_scorer(f1_score, pos_label=0),
        "ap": make_scorer(average_precision_score, pos_label=0),
        "auc": make_scorer(roc_auc_score),
    }
    scores = cross_validate(
        clf,
        dataset.x,
        dataset.y,
        cv=splitter,
        scoring=scoring,
        groups=groups,
        fit_params={"sample_weight": sample_weight, "groups": groups},
        n_jobs=5,
        verbose=1,
    )
    scores = {k[5:]: v for k, v in scores.items() if k.startswith("test_")}
    for key, vals in scores.items():
        print(f"{key}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")

    if save:
        clf.fit(dataset.x, dataset.y, sample_weight=sample_weight, groups=groups)
        save.parent.mkdir(parents=True, exist_ok=True)
        with open(save, "wb") as fid:
            pickle.dump(clf.best_estimator_, fid)
            print(f"Saved classifier to {save}")


if __name__ == "__main__":
    main()
