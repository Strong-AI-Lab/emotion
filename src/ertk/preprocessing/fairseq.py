"""Fairseq processor."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import MISSING, OmegaConf

from ertk.config import ERTKConfig
from ertk.utils import PathOrStr, is_mono_audio

from ._base import AudioClipProcessor, FeatureExtractor

__all__ = ["FairseqExtractorConfig", "FairseqExtractor"]


class Agg(Enum):
    """Aggregation method."""

    MEAN = "mean"
    """Mean of the output sequence."""
    MAX = "max"
    """Max of the output sequence."""
    NONE = "none"
    """Return entire sequence."""


@dataclass
class FairseqExtractorConfig(ERTKConfig):
    """Fairseq feature extractor configuration."""

    model_type: str = MISSING
    """Model type."""
    checkpoint: str = MISSING
    """Path to model checkpoint."""
    layer: str = "context"
    """Layer to extract features from."""
    aggregate: Agg = Agg.MEAN
    """Aggregation method."""
    device: str = "cuda"
    """Device to run model on."""
    arg_overrides: Dict[str, Any] = field(default_factory=dict)
    """Overrides for model arguments."""
    vq_path: Optional[str] = None
    """Path to vector quantiser."""
    vq_ids: bool = False
    """Whether to return VQ cluster ids."""
    vq_ids_as_string: bool = True
    """Whether to return VQ cluster ids as a single string of integers
    separated by spaces.
    """
    max_input_len: int = 1500000
    """Maximum input length."""
    mms_lang: str = "eng"
    """Language to use for MMS models."""


class FairseqExtractor(
    FeatureExtractor, AudioClipProcessor, fname="fairseq", config=FairseqExtractorConfig
):
    """Fairseq processor."""

    config: FairseqExtractorConfig

    def __init__(self, config: FairseqExtractorConfig) -> None:
        super().__init__(config)

        print(f"Loading model from {config.checkpoint}")
        if self.config.model_type == "data2vec":
            # Have to do everything manually since the released model is
            # 'experimental'
            import fairseq.examples.data2vec.models.data2vec_audio  # noqa
            from fairseq import models, tasks

            state = torch.load(config.checkpoint)
            cfg = OmegaConf.create(state["cfg"])

            del cfg.task["precompute_mask_indices"]
            del cfg.task["inferred_w2v_config"]
            task = tasks.setup_task(cfg.task)

            _model_dataclass = models.MODEL_DATACLASS_REGISTRY["data2vec_audio"]
            for key in set(cfg.model):
                if key not in _model_dataclass.__dataclass_fields__:
                    del cfg.model[key]
            model = task.build_model(cfg.model)

            del state["model"]["_ema"]
            state["model"]["final_proj.weight"] = state["model"].pop(
                "final_proj.0.weight"
            )
            state["model"]["final_proj.bias"] = state["model"].pop("final_proj.0.bias")
            model.load_state_dict(state["model"])

            args = OmegaConf.create(state["args"])
        elif self.config.model_type == "mms":
            from fairseq.checkpoint_utils import load_checkpoint_to_cpu
            from fairseq.tasks import setup_task

            ckpt = load_checkpoint_to_cpu(config.checkpoint)
            assert config.mms_lang in ckpt["adapter"]

            cfg = ckpt["cfg"]
            args = cfg
            task = setup_task(cfg.task, from_checkpoint=True)
            task.load_state_dict(ckpt["task_state"])  # Needed to build model
            model = task.build_model(cfg.model, from_checkpoint=True)
            model.load_state_dict(ckpt["model"], strict=True, model_cfg=cfg.model)

            # Load the correct language adapter
            ft_obj = ckpt["adapter"][config.mms_lang]
            ft_model = ft_obj["model"]
            cdevice = model.w2v_encoder.proj.weight.device
            cdtype = model.w2v_encoder.proj.weight.dtype
            proj_out, proj_in = ft_model["w2v_encoder.proj.weight"].shape
            ft_proj = torch.nn.Linear(proj_in, proj_out, bias=True)
            ft_proj.to(device=cdevice, dtype=cdtype)
            model.w2v_encoder.proj = ft_proj
            with torch.no_grad():
                for kk, vv in model.named_parameters():
                    if kk in ft_model:
                        vv.copy_(ft_model[kk])
            task.load_state_dict(ft_obj["task_state"])
        else:
            from fairseq.checkpoint_utils import load_model_ensemble_and_task

            [model], args, task = load_model_ensemble_and_task(
                [config.checkpoint], arg_overrides=config.arg_overrides
            )
        model.to(device=config.device)
        model.eval()
        self.model = model
        self.args = args
        self.task = task
        torch.set_grad_enabled(False)

        self.vq = None
        if config.vq_path:
            self.vq = joblib.load(config.vq_path)

    def process_instance(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if kwargs.pop("sr") != 16000:
            raise ValueError("Sample rate should be 16kHz")

        x = np.squeeze(x)
        if not is_mono_audio(x):
            raise ValueError("Audio must be mono")
        x = x[: self.config.max_input_len]
        tensor = torch.tensor(
            x, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)
        if hasattr(self.task, "normalize") and self.task.normalize:
            tensor = F.layer_norm(tensor, tensor.shape)

        layer: Union[str, int]
        try:
            layer = int(self.config.layer)
        except ValueError:
            layer = self.config.layer
        if layer == "context":
            layer = -1

        # Automatic speech recognition
        if self.config.model_type == "mms":
            outputs = self.model.forward(
                source=tensor, padding_mask=None, corpus_key=[0]
            )
            logprobs = self.model.get_normalized_probs(outputs, log_probs=True)
            toks = logprobs.transpose(0, 1).argmax(dim=-1).cpu().unique_consecutive()
            toks = toks[toks != 0]
            sent: str = self.task.target_dictionary.string(toks.long())
            sent = sent.replace(" ", "").replace("|", " ").strip()
            return np.array([sent], dtype=object)

        # Embeddings
        if self.config.model_type == "wav2vec":
            feats = self.model.feature_extractor(tensor)
            if self.model.vector_quantizer is not None:
                feats, _ = self.model.vector_quantizer.forward_idx(feats)
            if layer == -1:
                feats = self.model.feature_aggregator(feats)
        elif self.config.model_type in ["wav2vec2", "data2vec"]:
            if layer == "encoder":
                feats = self.model.feature_extractor(tensor)
            else:
                c = self.model.extract_features(tensor, None, layer=layer)["x"]
                # Transpose to (batch, feats, steps)
                feats = c.transpose(-2, -1)
        elif self.config.model_type == "hubert":
            if layer == "encoder":
                feats = self.model.feature_extractor(tensor)
            else:
                c, _ = self.model.extract_features(tensor, output_layer=layer)
                # Transpose to (batch, feats, steps)
                feats = c.transpose(-2, -1)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        if self.config.aggregate == Agg.NONE:
            feats = feats[0].transpose(1, 0).cpu().numpy()
        elif self.config.aggregate == Agg.MEAN:
            feats = feats[0].mean(-1).cpu().numpy()
        elif self.config.aggregate == Agg.MAX:
            feats = feats[0].max(-1).cpu().numpy()
        else:
            raise ValueError(f"Aggregation type {self.config.aggregate} not supported")

        if self.vq is None:
            return feats

        # Vector quantization
        if self.config.vq_ids:
            feats = self.vq.predict(feats)
            if self.config.vq_ids_as_string:
                feats = np.array([" ".join(map(str, feats))])
            feats = np.expand_dims(feats, -1)
        else:
            feats = self.vq.cluster_centers_[self.vq.predict(feats)]
        return feats

    def process_file(self, path: PathOrStr, sr: Optional[float] = None) -> np.ndarray:
        # Require 16 kHz sample rate
        return super().process_file(path, 16000)

    @property
    def dim(self) -> int:
        if self.config.vq_ids or self.config.model_type == "mms":
            return 1
        if self.config.layer == "encoder":
            return 512
        if self.config.model_type == "wav2vec":
            return 512
        elif self.config.model_type in ["wav2vec2", "data2vec", "hubert"]:
            return self.model.encoder.embedding_dim
        raise ValueError("Unknown dim")

    @property
    def is_sequence(self) -> bool:
        return self.config.aggregate == Agg.NONE

    @property
    def feature_names(self) -> List[str]:
        if self.config.vq_ids:
            return [f"{self.config.model_type}_vq_tok"]
        if self.config.model_type == "mms":
            return ["text"]
        return [f"{self.config.model_type}_{i}" for i in range(self.dim)]
