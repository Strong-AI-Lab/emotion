import pickle
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from emorec.dataset import CombinedDataset, LabelledDataset
from emorec.sklearn.models import PrecomputedSVC
from emorec.utils import PathlibPath
from sklearn.metrics import (average_precision_score, f1_score, get_scorer,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneGroupOut,
                                     cross_validate)


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False),
                nargs=-1)
@click.option('--save', type=Path, help="Location to save the model.")
@click.option('--cv', type=click.Choice(['speaker', 'corpus']),
              default='speaker', help="Cross-validation method.")
@click.option('--norm', type=click.Choice(['speaker', 'corpus']),
              default='speaker', help="Normalisation method.")
def main(input: Tuple[Path], save: Path, cv: str, norm: str):
    """Trains a model on the given INPUT models. Optionally saves the
    model as a pickle to the location specified by --save.
    """

    if len(input) == 0:
        raise ValueError("No input files specified.")

    dataset = CombinedDataset(*(LabelledDataset(path) for path in input))
    emotion_map = {x: 'emotional' for x in dataset.classes}
    emotion_map['neutral'] = 'neutral'
    dataset.map_classes(emotion_map)
    dataset.remove_classes(['emotional', 'neutral'])
    print(dataset.class_counts)

    dataset.normalise(scheme=norm)

    splitter = LeaveOneGroupOut()
    groups = None
    if cv == 'speaker':
        groups = dataset.speaker_group_indices
        if len(dataset.speakers) > 10:
            splitter = GroupKFold(6)
        print("Using speaker-independent cross-validation.")
    elif cv == 'corpus':
        groups = dataset.corpus_indices
        print("Using corpus-independent cross-validation.")
    else:
        splitter = KFold(10)

    class_weight = (dataset.n_instances
                    / (dataset.n_classes * dataset.class_counts))
    sample_weight = class_weight[dataset.y]

    scoring = {
        'war': get_scorer('accuracy'),
        'uar': get_scorer('balanced_accuracy'),
        'recall': make_scorer(recall_score, pos_label=0),
        'precision': make_scorer(precision_score, pos_label=0),
        'f1': make_scorer(f1_score, pos_label=0),
        'ap': make_scorer(average_precision_score, pos_label=0)
    }

    clf = PrecomputedSVC(C=1.0, kernel='rbf', gamma=2**-6, probability=True)
    scores = cross_validate(
        clf, dataset.x, dataset.y, cv=splitter, scoring=scoring, groups=groups,
        fit_params={'sample_weight': sample_weight}, n_jobs=6, verbose=0
    )

    mean_scores = {k[5:]: np.mean(v) for k, v in scores.items()
                   if k.startswith('test_')}

    print(f"Accuracy: {mean_scores['war']:.3f}")
    print(f"Bal. accuracy: {mean_scores['uar']:.3f}")
    print(f"Emotion recall: {mean_scores['recall']:.3f}")
    print(f"Emotion precision: {mean_scores['precision']:.3f}")
    print(f"F1 score: {mean_scores['f1']:.3f}")
    print(f"AP: {mean_scores['ap']:.3f}")

    if save:
        clf.fit(dataset.x, dataset.y, sample_weight=sample_weight)
        save.parent.mkdir(parents=True, exist_ok=True)
        with open(save, 'wb') as fid:
            pickle.dump(clf, fid)
            print(f"Saved classifier to {save}")


if __name__ == "__main__":
    main()
