from pathlib import Path

from constants import (
    corpus_name,
    feature_names,
    features_2d,
    features_3d,
    features_vlen,
    slices_vlen,
    test_data_dir,
)

from ertk.dataset import write_features


def main():
    assert test_data_dir.exists()

    with open(test_data_dir / "all_clips.txt") as fid:
        names = [Path(x.strip()).stem for x in fid]

    for format in ["arff", "csv", "nc"]:
        write_features(
            test_data_dir / f"features/features_2d.{format}",
            features_2d,
            names,
            corpus=corpus_name,
            feature_names=feature_names,
        )
        write_features(
            test_data_dir / f"features/features_3d.{format}",
            features_3d,
            names,
            corpus=corpus_name,
            feature_names=feature_names,
        )
        write_features(
            test_data_dir / f"features/features_vlen.{format}",
            features_vlen,
            names,
            corpus=corpus_name,
            feature_names=feature_names,
            slices=slices_vlen,
        )


if __name__ == "__main__":
    main()
