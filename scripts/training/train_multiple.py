import argparse
import pickle
from pathlib import Path

import numpy as np
from emorec.classification import PrecomputedSVC
from emorec.dataset import CombinedDataset, LabelledDataset
from sklearn.metrics import (average_precision_score, f1_score, get_scorer,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut, KFold,
                                     cross_validate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, nargs='+', help="Input datasets.")
    parser.add_argument(
        '--cv', type=str, default='speaker',
        help="Cross-validation method. One of {speaker, corpus}."
    )
    parser.add_argument('--norm', type=str, default='speaker',
                        help="Normalisation method. One of {speaker, corpus}.")
    parser.add_argument('--save', type=Path, help="Path to save model.")
    args = parser.parse_args()

    dataset = CombinedDataset(*(LabelledDataset(path) for path in args.input))
    emotion_map = {x: 'emotional' for x in dataset.classes}
    emotion_map['neutral'] = 'neutral'
    dataset.map_classes(emotion_map)
    dataset.remove_classes(['emotional', 'neutral'])
    print(dataset.class_counts)

    dataset.normalise(scheme=args.norm)

    cv = LeaveOneGroupOut()
    if args.cv == 'speaker':
        groups = dataset.speaker_group_indices
        if len(dataset.speakers) > 10:
            cv = GroupKFold(6)
        print("Using speaker-independent cross-validation.")
    elif args.cv == 'corpus':
        groups = dataset.corpus_indices
        print("Using corpus-independent cross-validation.")
    else:
        groups = None
        cv = KFold(10)

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
        clf, dataset.x, dataset.y, cv=cv, scoring=scoring, groups=groups,
        fit_params={'sample_weight': sample_weight}, n_jobs=6, verbose=0
    )

    mean_scores = {k[5:]: np.mean(v) for k, v in scores.items()
                   if k.startswith('test_')}

    print('Accuracy: {:.3f}'.format(mean_scores['war']))
    print('Bal. accuracy: {:.3f}'.format(mean_scores['uar']))
    print('Emotion recall: {:.3f}'.format(mean_scores['recall']))
    print('Emotion precision: {:.3f}'.format(mean_scores['precision']))
    print('F1 score: {:.3f}'.format(mean_scores['f1']))
    print('AP: {:.3f}'.format(mean_scores['ap']))

    if args.save:
        clf.fit(dataset.x, dataset.y, sample_weight=sample_weight)
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, 'wb') as fid:
            pickle.dump(clf, fid)
            print("Saved classifier to {}".format(args.save))


if __name__ == "__main__":
    main()
