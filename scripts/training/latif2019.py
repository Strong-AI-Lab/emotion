"""Implementation of [1].

[1] S. Latif, R. Rana, S. Khalifa, R. Jurdak, and J. Epps, "Direct
Modelling of Speech Emotion from Raw Speech," in Interspeech 2019, Graz,
Sep. 2019, pp. 3920–3924, doi: 10.21437/Interspeech.2019-3252.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import get_scorer, make_scorer, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop

from emorec.dataset import LabelledDataset
from emorec.tensorflow.classification import (
    BalancedSparseCategoricalAccuracy,
    tf_train_val_test,
)
from emorec.tensorflow.models.latif2019 import model as _model
from emorec.tensorflow.utils import create_tf_dataset_ragged


def test_corpus(corpus: str):
    cv = LeaveOneGroupOut()
    reps = 1

    dataset = LabelledDataset(f"datasets/{corpus}/files.txt")
    dataset.pad_arrays()
    dataset.clip_arrays

    def model_fn():
        model = _model(dataset.n_classes)
        model.compile(
            optimizer=RMSprop(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=[
                SparseCategoricalAccuracy(name="war"),
                BalancedSparseCategoricalAccuracy(name="uar"),
            ],
        )
        return model

    metrics = (
        ["uar", "war"]
        + [x + "_rec" for x in dataset.classes]
        + [x + "_prec" for x in dataset.classes]
    )
    df = pd.DataFrame(
        index=pd.RangeIndex(1, reps + 1, name="rep"), columns=metrics + ["params"]
    )

    scoring = {"war": get_scorer("accuracy"), "uar": get_scorer("balanced_accuracy")}
    for i, c in enumerate(dataset.classes):
        scoring.update(
            {
                c + "_rec": make_scorer(recall_score, average=None, labels=[i]),
                c + "_prec": make_scorer(precision_score, average=None, labels=[i]),
            }
        )

    for rep in range(1, reps + 1):
        print(f"Rep {rep}/{reps}")
        fold = 1
        scores = defaultdict(list)
        for train, _test in cv.split(
            dataset.x, dataset.y, dataset.speaker_group_indices
        ):
            train_data = create_tf_dataset_ragged(
                dataset.x[train], dataset.y[train], batch_size=64, shuffle=True
            )
            _test_x = dataset.x[_test]
            _test_y = dataset.y[_test]
            for valid, test in cv.split(
                _test_x, _test_y, dataset.speaker_indices[_test]
            ):
                print(f"Fold {fold}/{len(dataset.speakers)}")
                callbacks = (
                    [
                        EarlyStopping(
                            monitor="val_uar",
                            patience=20,
                            restore_best_weights=True,
                            mode="max",
                        ),
                        ReduceLROnPlateau(
                            monitor="val_uar", factor=0.5, patience=5, mode="max"
                        ),
                    ],
                )
                valid_data = create_tf_dataset_ragged(
                    _test_x[valid], _test_y[valid], batch_size=64, shuffle=True
                )
                test_data = create_tf_dataset_ragged(
                    _test_x[test], _test_y[test], batch_size=64, shuffle=False
                )
                _scores = tf_train_val_test(
                    model_fn,
                    train_data=train_data,
                    valid_data=valid_data,
                    test_data=test_data,
                    scoring=scoring,
                    callbacks=callbacks,
                    epochs=100,
                    verbose=0,
                )
                for k in _scores:
                    scores[k].append(_scores[k])
                n_epochs = len(_scores["history"]["loss"])
                print(f"Fold {fold} finished after {n_epochs} epochs.")
                fold += 1

        mean_scores = {
            k[5:]: np.mean(v) for k, v in scores.items() if k.startswith("test_")
        }
        war = mean_scores["war"]
        uar = mean_scores["uar"]
        recall = tuple(mean_scores[c + "_rec"] for c in dataset.classes)
        precision = tuple(mean_scores[c + "_prec"] for c in dataset.classes)

        df.loc[rep, "war"] = war
        df.loc[rep, "uar"] = uar
        for i, c in enumerate(dataset.classes):
            df.loc[rep, c + "_rec"] = recall[i]
            df.loc[rep, c + "_prec"] = precision[i]
    output_dir = Path("results") / "latif2019" / corpus / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "raw_audio.csv")
    print(f"Wrote CSV to {output_dir / 'raw_audio.csv'}")


def main():
    tf.get_logger().setLevel(40)  # ERROR level
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Print model structure summary
    model = _model(4)
    model.summary()
    del model
    tf.keras.backend.clear_session()

    test_corpus("iemocap")
    test_corpus("msp-improv")


if __name__ == "__main__":
    main()
