import numpy as np
import omegaconf
import pytest

from ertk.dataset.dataset import CombinedDataset, Dataset, load_multiple

from .constants import (
    all_clips_names,
    all_clips_unsorted_names,
    feature_names,
    features_2d,
    features_3d,
    subset_names,
    test_data_dir,
)


class TestDataset:
    def test_default_subset(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert data.subset == "all"

    def test_sorted(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", subset="all")
        assert data.corpus == "test_corpus"
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert len(data.annotations["speaker"]) == 12
        assert len(data.get_annotations("speaker")) == len(all_clips_names)
        assert np.array_equal(data.speakers, data.get_annotations("speaker"))
        assert data.partitions == {"label", "speaker", "corpus", "_audio_path"}
        assert data.n_speakers == len(data.speaker_names)
        assert data.n_speakers == 2
        assert data.speaker_names == ["1001", "1002"]
        assert len(data.names) == len(data)
        assert np.array_equal(data.names, all_clips_names)
        assert data.subset == "all"
        assert data.subsets.keys() == {"all", "subset1", "all_unsorted"}
        assert list(data.speaker_counts) == [6, 6]
        assert list(data.speaker_indices) == [0] * 6 + [1] * 6
        assert np.array_equal(data.speaker_counts, data.get_group_counts("speaker"))
        assert np.array_equal(data.speaker_indices, data.get_group_indices("speaker"))

    def test_unsorted(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", subset="all_unsorted")
        assert np.array_equal(data.names, all_clips_unsorted_names)
        assert list(data.speaker_counts) == [6, 6]
        assert data.subset == "all_unsorted"

    def test_subset(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", subset="subset1")
        assert data.corpus == "test_corpus"
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert len(data._annotations["speaker"]) == 12
        assert len(data.annotations["speaker"]) == 6
        assert len(data.get_annotations("speaker")) == len(subset_names)
        assert data.partitions == {"label", "speaker", "corpus", "_audio_path"}
        assert data.n_speakers == len(data.speaker_names)
        assert data.n_speakers == 2
        assert set(data.speaker_names) == {"1001", "1002"}
        assert len(data.names) == len(data)
        assert np.array_equal(data.names, subset_names)
        assert data.subset == "subset1"
        assert data.subsets.keys() == {"all", "subset1", "all_unsorted"}

    def test_dataset2(self):
        data = Dataset(test_data_dir / "corpus2.yaml", subset="all")
        assert data.corpus == "test_corpus2"
        assert set(data.annotations.columns) == {
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert data.partitions == {"label", "speaker", "corpus", "_audio_path"}
        assert data.subset == "all"
        assert data.subsets.keys() == {"all"}

    def test_bad_corpus_info(self):
        with pytest.raises(omegaconf.errors.ValidationError):
            Dataset(test_data_dir / "corpus_fail.yaml")

    def test_features_2d(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        assert np.allclose(data.x, features_2d)
        assert data.feature_names == feature_names
        assert data.n_features == len(data.feature_names)
        assert data.n_instances == len(data) == len(all_clips_names)
        assert np.array_equal(data.names, all_clips_names)

    def test_features_3d(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_3d")
        assert np.allclose(data.x, features_3d)
        assert data.feature_names == feature_names
        assert data.n_instances == len(all_clips_names)
        assert np.array_equal(data.names, all_clips_names)

    def test_copy(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        data_copy = data.copy()
        assert data_copy.corpus == data.corpus
        assert data_copy.annotations.equals(data.annotations)
        assert data_copy.annotations is not data.annotations
        assert data_copy.subset == data.subset
        assert np.array_equal(data_copy.names, data.names)
        assert data_copy.names is not data.names
        assert data_copy.partitions == data.partitions
        assert data_copy.partitions is not data.partitions
        assert np.array_equal(data_copy.x, data.x)
        assert data_copy.x is not data.x

    def test_remove_partition(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        data.remove_annotation("label")
        assert set(data.annotations.columns) == {
            "annot1",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert data.partitions == {"speaker", "corpus", "_audio_path"}

    def test_remove_annotation(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        data.remove_annotation("annot1")
        assert set(data.annotations.columns) == {
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert data.partitions == {"label", "speaker", "corpus", "_audio_path"}

    def test_rename_partition(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        data.rename_annotation("annot1", "annot2")
        assert set(data.annotations.columns) == {
            "annot2",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert data.partitions == {"label", "speaker", "corpus", "_audio_path"}

    def test_rename_annotation(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "corpus",
            "_audio_path",
        }
        data.rename_annotation("label", "xyz")
        assert set(data.annotations.columns) == {
            "annot1",
            "xyz",
            "speaker",
            "corpus",
            "_audio_path",
        }
        assert data.partitions == {"xyz", "speaker", "corpus", "_audio_path"}

    def test_update_annotation(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        with pytest.raises(RuntimeError):
            data.update_annotation("speaker", {"1001_DFA_HAP_XX": "xyz"})

    def test_add_annotation(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        new_annot = list("cbaabcbccaaa")
        data.update_annotation("new_annot", new_annot)
        assert data.partitions == {
            "label",
            "speaker",
            "new_annot",
            "corpus",
            "_audio_path",
        }
        assert set(data.annotations.columns) == {
            "annot1",
            "label",
            "speaker",
            "new_annot",
            "corpus",
            "_audio_path",
        }
        assert np.array_equal(data.get_annotations("new_annot"), new_annot)
        assert data.get_group_names("new_annot") == ["a", "b", "c"]
        assert list(data.get_group_counts("new_annot")) == [5, 3, 4]
        indices = [2, 1, 0, 0, 1, 2, 1, 2, 2, 0, 0, 0]
        assert list(data.get_group_indices("new_annot")) == indices

    def test_map_groups(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        data.map_groups("speaker", {"1001": "xyz"})
        assert data.speaker_names == ["1002", "xyz"]
        assert list(data.speaker_counts) == [6, 6]

    def test_modify_annotations(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        new = data.annotations["speaker"].cat.add_categories(["xyz"])
        new["1001_DFA_HAP_XX"] = "xyz"
        data.update_annotation("speaker", new)
        assert data.speaker_names == ["1001", "1002", "xyz"]
        assert list(data.speaker_counts) == [5, 6, 1]

    def test_drop_instances_no_features(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        data.remove_instances(drop=["1001_DFA_HAP_XX", "1001_DFA_HAP_XX"])
        ref_names = all_clips_names.copy()
        ref_names.remove("1001_DFA_HAP_XX")
        assert np.array_equal(data.names, ref_names)
        assert list(data.speakers) == ["1001"] * 5 + ["1002"] * 6
        assert list(data.speaker_counts) == [5, 6]
        assert list(data.speaker_indices) == [0] * 5 + [1] * 6

    def test_drop_instances_features(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        data.remove_instances(drop=["1001_DFA_HAP_XX", "1001_DFA_HAP_XX"])
        assert len(data.x) == len(all_clips_names) - 1

    def test_keep_instances_no_features(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        data.remove_instances(keep=["1001_DFA_HAP_XX", "1001_DFA_NEU_XX"])
        assert list(data.names) == ["1001_DFA_HAP_XX", "1001_DFA_NEU_XX"]
        assert len(data.get_annotations("speaker")) == 2
        assert list(data.speakers) == ["1001", "1001"]
        assert data.speaker_names == ["1001"]
        assert list(data.speaker_counts) == [2]
        assert list(data.speaker_indices) == [0, 0]

    def test_keep_instances_features(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        data.remove_instances(keep=["1001_DFA_HAP_XX", "1001_DFA_NEU_XX"])
        assert len(data.x) == 2

    def test_drop_group(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        data.remove_groups("speaker", drop=["1001"])
        assert data.speaker_names == ["1002"]
        assert list(data.speakers) == ["1002"] * 6
        assert data.n_speakers == 1
        assert list(data.speaker_counts) == [6]
        assert list(data.speaker_indices) == [0] * 6

    def test_keep_group(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", features="features_2d")
        data.remove_groups("speaker", keep=["1001"])
        assert data.speaker_names == ["1001"]
        assert list(data.speakers) == ["1001"] * 6
        assert data.n_speakers == 1
        assert list(data.speaker_counts) == [6]
        assert list(data.speaker_indices) == [0] * 6

    def test_labels(self):
        data = Dataset(test_data_dir / "corpus_info.yaml")
        assert len(data.labels) == len(all_clips_names)
        assert len(data.classes) == 6
        assert data.n_classes == len(data.classes)
        assert np.array_equal(data.labels, data.get_annotations("label"))
        assert data.classes == data.get_group_names("label")
        assert data.classes == [
            "anger",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
        ]
        assert list(data.class_counts) == [2] * 6
        assert list(data.y) == [0, 1, 2, 3, 4, 5] * 2
        assert np.array_equal(data.class_counts, data.get_group_counts("label"))
        assert np.array_equal(data.y, data.get_group_indices("label"))

    def test_labels_subset(self):
        data = Dataset(test_data_dir / "corpus_info.yaml", subset="subset1")
        assert len(data.labels) == len(subset_names)
        assert data.n_classes == 3
        assert data.classes == ["anger", "fear", "happiness"]
        assert list(data.class_counts) == [2] * 3
        assert list(data.y) == [0, 1, 2] * 2


class TestCombinedDataset:
    def test_combined(self):
        data1 = Dataset(test_data_dir / "corpus_info.yaml")
        data2 = Dataset(test_data_dir / "corpus2.yaml")
        combined = CombinedDataset(data1, data2)
        assert combined.corpus == "combined"
        assert combined.corpus_names == ["test_corpus", "test_corpus2"]
        assert len(combined.names) == 24
        assert set(combined.annotations.columns) == {
            "corpus",
            "label",
            "speaker",
            "_audio_path",
        }
        assert list(combined.corpus_indices) == [0] * 12 + [1] * 12
        ref_names = [f"test_corpus_{x}" for x in all_clips_names]
        ref_names += [f"test_corpus2_{x}" for x in all_clips_names]
        assert np.array_equal(combined.names, ref_names)
        assert combined.speaker_names == [
            "test_corpus2_1001",
            "test_corpus2_1002",
            "test_corpus_1001",
            "test_corpus_1002",
        ]


def test_load_multiple():
    files = [test_data_dir / "corpus_info.yaml", test_data_dir / "corpus2.yaml"]
    combined = load_multiple(files, "features_2d")
    assert np.allclose(combined.x, np.r_[features_2d, features_2d])
    assert combined.corpus == "combined"
    assert combined.corpus_names == ["test_corpus", "test_corpus2"]
