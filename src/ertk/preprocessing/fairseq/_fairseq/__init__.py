# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.10.0"

import os
import sys

import torch

from . import meters

sys.modules["fairseq.meters"] = meters


def load_model(filename, arg_overrides=None):
    from .wav2vec import Wav2VecModel

    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))
    state = torch.load(filename, map_location="cpu")

    args = state["args"]
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    model = Wav2VecModel(args)
    model.load_state_dict(state["model"], strict=True)
    return model, args
