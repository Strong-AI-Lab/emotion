from pathlib import Path
from typing import List

import numpy as np
import pytest

from ertk.preprocessing import (
    audioset,
    fairseq,
    huggingface,
    keras_apps,
    opensmile,
    openxbow,
    phonemize,
    resample,
    spectrogram,
    speechbrain,
    vad_trim,
)

test_data_dir = Path(__file__).parent.parent / "dataset/test_data"
files = list(test_data_dir.glob("resampled/*.wav"))


@pytest.fixture(scope="module")
def audio() -> List[np.ndarray]:
    import librosa

    return [librosa.load(x, sr=16000, mono=True)[0] for x in files]


@pytest.fixture(scope="module")
def transcript() -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(test_data_dir / "transcript.csv", index_col=0)
    return df.to_numpy()


AUDIOSET_DIR = "/mnt/storage2/models/audioset"
FAIRSEQ_DIR = "/mnt/storage2/models/fairseq"


class TestAudioset:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_vggish(self, audio):
        config = audioset.VGGishExtractorConfig(
            model_dir=f"{AUDIOSET_DIR}/vggish", postprocess=True
        )
        ext = audioset.VGGishExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 128
        assert all(x.shape == (ext.dim,) for x in feats)
        assert not ext.is_sequence

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_yamnet(self, audio):
        config = audioset.YamnetExtractorConfig(model_dir=f"{AUDIOSET_DIR}/yamnet")
        ext = audioset.YamnetExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1024
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestFairseq:
    @pytest.mark.parametrize("layer", ["context", "encoder"])
    @pytest.mark.parametrize(
        "checkpoint",
        [
            f"{FAIRSEQ_DIR}/wav2vec/wav2vec_large.pt",
            f"{FAIRSEQ_DIR}/wav2vec/vq-wav2vec.pt",
        ],
    )
    def test_fairseq_w2v(self, layer, checkpoint, audio):
        config = fairseq.FairseqExtractorConfig(
            model_type="wav2vec", checkpoint=checkpoint, layer=layer
        )
        ext = fairseq.FairseqExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 512
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)

    @pytest.mark.parametrize(
        ["model_type", "checkpoint", "layer", "dim"],
        [
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/libri960_big.pt", "context", 1024),
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/wav2vec_small.pt", "context", 768),
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/xlsr_53_56k.pt", "context", 1024),
            ("data2vec", f"{FAIRSEQ_DIR}/data2vec/audio_base_ls.pt", "context", 768),
            ("hubert", f"{FAIRSEQ_DIR}/hubert/hubert_base_ls960.pt", "context", 768),
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/libri960_big.pt", "encoder", 512),
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/wav2vec_small.pt", "encoder", 512),
            ("wav2vec2", f"{FAIRSEQ_DIR}/wav2vec2/xlsr_53_56k.pt", "encoder", 512),
            ("data2vec", f"{FAIRSEQ_DIR}/data2vec/audio_base_ls.pt", "encoder", 512),
            ("hubert", f"{FAIRSEQ_DIR}/hubert/hubert_base_ls960.pt", "encoder", 512),
        ],
    )
    def test_fairseq_w2v2(self, model_type, checkpoint, layer, dim, audio):
        config = fairseq.FairseqExtractorConfig(
            model_type=model_type, checkpoint=checkpoint, layer=layer
        )
        ext = fairseq.FairseqExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == dim
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)

    def test_fairseq_hubert_vq(self, audio):
        config = fairseq.FairseqExtractorConfig(
            model_type="hubert",
            checkpoint=f"{FAIRSEQ_DIR}/hubert/hubert_base_ls960.pt",
            layer="context",
            aggregate=fairseq.Agg.NONE,
            vq_path=f"{FAIRSEQ_DIR}/hubert/km200/km.bin",
            vq_ids=True,
            vq_ids_as_string=False,
        )
        ext = fairseq.FairseqExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.is_sequence
        assert ext.dim == 1
        assert all(x.shape[1] == ext.dim for x in feats)


class TestHuggingFace:
    def test_hf_w2v2(self, audio):
        config = huggingface.HuggingFaceExtractorConfig(
            model="facebook/wav2vec2-base-960h", task="EMBEDDINGS", layer="context"
        )
        ext = huggingface.HuggingFaceExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(audio)
        assert ext.dim == 768
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestKerasApps:
    @pytest.mark.parametrize("model", ["vgg16", "vgg19", "resnet50", "inception_v3"])
    def test_keras_apps(self, model, audio):
        config = keras_apps.KerasAppsExtractorConfig(model=model)
        ext = keras_apps.KerasAppsExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestOpenSMILE:
    def test_eGeMAPS(self, audio):
        config = opensmile.OpenSMILEExtractorConfig(opensmile_config="eGeMAPS")
        ext = opensmile.OpenSMILEExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 88
        assert not ext.is_sequence
        assert all(x.shape == (1, ext.dim) for x in feats)

    def test_eGeMAPS_lld(self, audio):
        config = opensmile.OpenSMILEExtractorConfig(
            opensmile_config="eGeMAPS", levels=["lld"]
        )
        ext = opensmile.OpenSMILEExtractor(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 23
        assert ext.is_sequence
        assert all(x.shape[1] == ext.dim for x in feats)


class TestOpenXBOW:
    def test_openxbow(self):
        config = openxbow.OpenXBOWExtractorConfig(
            xbowargs=["-a=10", "-size=100", "-norm=1"]
        )
        xs = np.random.randn(10, 100, 50)
        ext = openxbow.OpenXBOWExtractor(config)
        feats = list(ext.process_instances(xs, batch_size=-1, sr=16000))
        assert len(feats) == 10
        assert ext.dim == 100
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestPhonemize:
    def test_phonemize(self, transcript):
        config = phonemize.PhonemizeConfig(language="en-us", backend="festival")
        ext = phonemize.Phonemizer(config)
        feats = list(ext.process_instances(transcript, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)

    def test_phonemize_batch(self, transcript):
        config = phonemize.PhonemizeConfig(language="en-us", backend="festival")
        ext = phonemize.Phonemizer(config)
        feats = list(ext.process_instances(transcript, batch_size=-1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestResample:
    def test_resample(self, audio):
        config = resample.ResampleConfig(sample_rate=8000)
        ext = resample.Resampler(config)
        feats = list(ext.process_instances(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
