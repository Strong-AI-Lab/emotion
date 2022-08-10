import math
from typing import Any, Callable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Attention1D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channels, 1) / math.sqrt(in_channels))

    def forward(self, x: torch.Tensor):
        # x : (..., length, in_channels)
        alpha = F.softmax(x.matmul(self.weight), -2)
        # alpha : (..., length, 1)
        r = torch.sum(alpha * x, -2)
        return r


def get_activation(act: str, *args, **kwargs) -> nn.Module:
    """Get torch Module for activation function.

    Parameters
    ----------
    act: str
        String representing the activation function.
    *args, **kwargs: optional
        Arguments and keyword arguments to pass to activation function
        module `__init__()`.

    Returns
    -------
    torch.nn.Module
        The activation function as a `Module`.
    """
    cls = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "linear": nn.Identity,
        "tanh": nn.Tanh,
    }[act]
    return cls(*args, **kwargs)


def make_fc(
    dims: List[int],
    activation: str = "relu",
    activation_args: Sequence[Any] = [],
    init_fn: Callable = nn.init.xavier_uniform_,
    norm: str = "batch",
    dropout: float = 0.2,
):
    """Makes a fully connected module with the given numbers of hidden
    units, and given activation after every layer except the last.

    Parameters
    ----------
    dims: list of int
        Layer dimensions.
    activation: torch.nn.Module
        The activation to use as a module class.
    activation_param: optional
        A parameter to pass to the activation.
    init_fn: callable
        The weight initialisation scheme. Default is 'xavier_uniform'.
    norm: str
        Whether to apply normalisation after each layer. If "batch" then
        apply batch normalisation. If "layer" apply layer normalisation.

    Returns
    -------
    torch.nn.Module
        A `Module` which contains the corresponding layers, activations,
        and optional dropout and normalisation.
    """
    param = activation_args[0] if len(activation_args) > 0 else None
    init_gain = nn.init.calculate_gain(activation, param=param)

    module = nn.Sequential()
    for i in range(len(dims) - 2):
        linear = nn.Linear(dims[i], dims[i + 1])
        init_fn(linear.weight, gain=init_gain)
        module.append(linear)
        module.append(get_activation(activation, *activation_args))
        if norm == "batch":
            module.append(nn.BatchNorm1d(dims[i + 1]))
        elif norm == "layer":
            module.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            module.append(nn.Dropout(dropout))
    linear = nn.Linear(dims[-2], dims[-1])
    init_fn(linear.weight, gain=init_gain)
    module.append(nn.Linear(dims[-2], dims[-1]))
    return module
