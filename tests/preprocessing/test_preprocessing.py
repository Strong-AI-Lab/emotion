import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

from ertk.preprocessing import (
    audioset,
    encodec,
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

test_data_dir = Path(__file__).parent / "../test_data"
files = list(test_data_dir.glob("resampled/*.wav"))


@pytest.fixture(scope="module")
def audio() -> list[np.ndarray]:
    import librosa

    return [librosa.load(x, sr=16000, mono=True)[0] for x in files]


@pytest.fixture(scope="module")
def transcript() -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(test_data_dir / "transcript.csv", index_col=0)
    return df.to_numpy()


def installed(package: str) -> bool:
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def skip_missing_import(package: str):
    return pytest.mark.skipif(
        not installed(package), reason=f"{package} not installed."
    )


def skip_missing_path(path: str):
    return pytest.mark.skipif(
        not Path(path).exists(), reason=f"Path {path} doesn't exist."
    )


load_dotenv()
FAIRSEQ_DIR = os.getenv("FAIRSEQ_DIR")


@skip_missing_import("tensorflow_hub")
class TestAudioset:
    def test_vggish(self, audio):
        config = audioset.VGGishExtractorConfig()
        ext = audioset.VGGishExtractor(config)
        feats = list(ext.process_all(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 128
        assert all(x.shape == (ext.dim,) for x in feats)
        assert not ext.is_sequence

    def test_yamnet(self, audio):
        config = audioset.YAMNetExtractorConfig()
        ext = audioset.YAMNetExtractor(config)
        feats = list(ext.process_all(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1024
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


@skip_missing_import("encodec")
class TestEncodec:
    @pytest.mark.parametrize(
        ["model", "dim"],
        [(encodec.Model.ENCODEC_48kHz, 2048), (encodec.Model.ENCODEC_24kHz, 4096)],
    )
    @pytest.mark.parametrize("aggregate", [encodec.Agg.MEAN, encodec.Agg.MAX])
    def test_encodec(self, model, aggregate, dim, audio):
        config = encodec.EncodecExtractorConfig(model=model, aggregate=aggregate)
        ext = encodec.EncodecExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == dim
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)

    def test_encodec_vq(self, audio):
        config = encodec.EncodecExtractorConfig(
            model=encodec.Model.ENCODEC_48kHz, vq_ids=True, vq_ids_as_string=False
        )
        ext = encodec.EncodecExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 16
        assert ext.is_sequence
        assert all(x.shape[1] == ext.dim for x in feats)

    def test_encodec_vq_str(self, audio):
        config = encodec.EncodecExtractorConfig(
            model=encodec.Model.ENCODEC_48kHz, vq_ids=True, vq_ids_as_string=True
        )
        ext = encodec.EncodecExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert ext.is_sequence
        assert all(x.shape[1] == ext.dim for x in feats)


class TestFairseq:
    @skip_missing_path(f"{FAIRSEQ_DIR}/wav2vec")
    @pytest.mark.parametrize("layer", ["context", "encoder"])
    @pytest.mark.parametrize(
        "checkpoint",
        [
            f"{FAIRSEQ_DIR}/wav2vec/wav2vec_large.pt",
            f"{FAIRSEQ_DIR}/wav2vec/vq-wav2vec.pt",
        ],
    )
    def test_fairseq_w2v(self, layer, checkpoint, audio):
        config = fairseq.FairseqExtractorConfig(checkpoint=checkpoint, layer=layer)
        ext = fairseq.FairseqExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 512
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


@skip_missing_import("transformers")
class TestHuggingFace:
    def test_hf_w2v2(self, audio):
        config = huggingface.HuggingFaceExtractorConfig(
            model="facebook/wav2vec2-base-960h",
            task=huggingface.Task.EMBEDDINGS,
            layer="context",
        )
        ext = huggingface.HuggingFaceExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(audio)
        assert ext.dim == 768
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestKerasApps:
    @pytest.mark.parametrize("model", ["vgg16", "vgg19", "resnet50", "inception_v3"])
    def test_keras_apps(self, model, audio):
        config = keras_apps.KerasAppsExtractorConfig(model=model)
        ext = keras_apps.KerasAppsExtractor(config)
        feats = list(ext.process_all(audio, batch_size=32, sr=16000))
        assert len(feats) == len(files)
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


@skip_missing_import("opensmile")
class TestOpenSMILE:
    def test_eGeMAPS(self, audio):
        config = opensmile.OpenSMILEExtractorConfig(opensmile_config="eGeMAPSv02")
        ext = opensmile.OpenSMILEExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 88
        assert not ext.is_sequence
        assert all(x.shape == (1, ext.dim) for x in feats)

    def test_eGeMAPS_lld(self, audio):
        config = opensmile.OpenSMILEExtractorConfig(
            opensmile_config="eGeMAPSv02", levels=["lld"]
        )
        ext = opensmile.OpenSMILEExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 25
        assert ext.is_sequence
        assert all(x.shape[1] == ext.dim for x in feats)


class TestOpenXBOW:
    def test_openxbow(self):
        config = openxbow.OpenXBOWExtractorConfig(
            xbowargs=["-a=10", "-size=100", "-norm=1"]
        )
        xs = np.random.randn(10, 100, 50)
        ext = openxbow.OpenXBOWExtractor(config)
        feats = list(ext.process_all(xs, batch_size=-1, sr=16000))
        assert len(feats) == 10
        assert ext.dim == 100
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


@skip_missing_import("phonemizer.backend.festival.festival")
@pytest.mark.skipif(os.system("command -v festival") != 0, reason="festival not found")
class TestPhonemize:
    def test_phonemize(self, transcript):
        config = phonemize.PhonemizeConfig(language="en-us", backend="festival")
        ext = phonemize.Phonemizer(config)
        feats = list(ext.process_all(transcript, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)

    def test_phonemize_batch(self, transcript):
        config = phonemize.PhonemizeConfig(language="en-us", backend="festival")
        ext = phonemize.Phonemizer(config)
        feats = list(ext.process_all(transcript, batch_size=-1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


@skip_missing_import("resampy")
class TestResample:
    def test_resample(self, audio):
        config = resample.ResampleConfig(sample_rate=8000)
        ext = resample.Resampler(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)


class TestSpectrogram:
    def test_spectrogram(self, audio):
        config = spectrogram.SpectrogramExtractorConfig(
            window_size=0.025, window_shift=0.01, n_mels=80, fmin=0, fmax=8000
        )
        ext = spectrogram.SpectrogramExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 80
        assert ext.is_sequence
        assert all(x.shape[1] == ext.dim for x in feats)


@skip_missing_import("speechbrain")
class TestSpeechBrain:
    def test_speechbrain(self, audio):
        config = speechbrain.SpeechBrainExtractorConfig(
            model="speechbrain/asr-transformer-transformerlm-librispeech",
            task=speechbrain.Task.ASR,
        )
        ext = speechbrain.SpeechBrainExtractor(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
        assert ext.dim == 1
        assert not ext.is_sequence
        assert all(x.shape == (ext.dim,) for x in feats)


class TestVADTrim:
    def test_vad_trim(self, audio):
        config = vad_trim.VADTrimmerConfig()
        ext = vad_trim.VADTrimmer(config)
        feats = list(ext.process_all(audio, batch_size=1, sr=16000))
        assert len(feats) == len(files)
