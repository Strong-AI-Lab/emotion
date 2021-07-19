import pickle
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
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

from emorec.dataset import load_multiple
from emorec.sklearn.models import PrecomputedSVC
from emorec.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.argument("--features", required=True, help="Features to load.")
@click.option("--save", type=Path, help="Location to save the model.")
@click.option(
    "--cv",
    type=click.Choice(["speaker", "corpus"]),
    help="Cross-validation method.",
)
@click.option(
    "--normalise",
    type=click.Choice(["speaker", "corpus", "online"]),
    default="speaker",
    help="Normalisation method.",
)
def main(input: Tuple[Path], features: str, save: Path, cv: str, normalise: str):
    """Trains a model on the given INPUT datasets. INPUT files must be
    features files.

    Optionally pickles the model.
    """

    dataset = load_multiple(input, features)

    print("Loaded all datasets and labels.")
    emotion_map = {x: "emotional" for x in dataset.classes}
    emotion_map["neutral"] = "neutral"
    dataset.map_classes(emotion_map)
    dataset.remove_classes(keep={"emotional", "neutral"})
    dataset.normalise(partition=normalise)
    print(dataset)

    splitter = LeaveOneGroupOut()
    groups = None
    if cv == "speaker" or cv is None:
        groups = dataset.speaker_indices
        if len(dataset.speaker_names) > 10:
            splitter = GroupKFold(5)
        print("Using speaker-independent cross-validation.")
    elif cv == "corpus":
        groups = dataset.corpus_indices
        print("Using corpus-independent cross-validation.")
    else:
        splitter = KFold(5)

    class_weight = dataset.n_instances / (dataset.n_classes * dataset.class_counts)
    sample_weight = class_weight[dataset.y]

    param_grid = {
        "C": 2.0 ** np.arange(-6, 7, 2),
        "gamma": 2.0 ** np.arange(-12, -1, 2),
    }
    clf = GridSearchCV(
        PrecomputedSVC(kernel="rbf", probability=True),
        param_grid,
        scoring="balanced_accuracy",
        cv=GroupKFold(2),
        n_jobs=10 if cv is None else 2,
        verbose=1 if cv is None else 0,
    )

    if cv is not None:
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
        scores_df = pd.DataFrame(scores)
        print(scores_df)
        print(scores_df.mean())

    if save:
        clf.fit(dataset.x, dataset.y, sample_weight=sample_weight, groups=groups)
        save.parent.mkdir(parents=True, exist_ok=True)
        with open(save, "wb") as fid:
            pickle.dump(clf.best_estimator_, fid)
            print(f"Saved classifier to {save}")


if __name__ == "__main__":
    main()
