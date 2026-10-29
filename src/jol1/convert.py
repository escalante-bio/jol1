"""Torch -> JAX weight conversion. The only module that imports torch (``reference`` extra)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import jax.numpy as jnp
import torch

# The reference picks its LayerNorm implementation at import time; the fused CUDA kernel
# needs a GPU build of torch. Both share parameters and math, and both convert.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import jol1.confidence  # noqa: E402, F401  (registers classes)
import jol1.diffusion  # noqa: E402, F401
import jol1.embedders  # noqa: E402, F401
import jol1.model  # noqa: E402, F401
import jol1.nn  # noqa: E402, F401
import jol1.pairformer  # noqa: E402, F401
import jol1.transformer  # noqa: E402, F401
from jol1.backend import REGISTRY, from_torch  # noqa: E402
from jol1.model import Mol1  # noqa: E402


def _resolve(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


for _path, _convert in REGISTRY:
    from_torch.register(_resolve(_path), _convert)
from_torch.register(torch.Tensor, lambda t: jnp.array(t.detach().cpu().float().numpy()))
from_torch.register(torch.nn.ModuleList, lambda m: tuple(from_torch(child) for child in m))


def inference_configs(model_name: str = "mol-1-base"):
    """The reference's ``mol pred`` configuration (model, sampler and data settings), with
    the plain-PyTorch triangle kernels (all kernels share weights and math)."""
    import copy

    from mol.config.base import configs as base_configs
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs as infer_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    configs = copy.deepcopy({**base_configs, "data": data_configs, **infer_configs})
    deep_update(configs, model_configs[model_name])
    configs = parse_configs(configs, arg_str=None, fill_required_with_null=True)
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    return configs


def load_reference_model(checkpoint_path: Path) -> torch.nn.Module:
    """The reference ``Mol`` model with the checkpoint's weights, on CPU in eval mode."""
    from unittest import mock

    import numpy as np
    from mol.model.mol import Mol
    from mol.model.triangular import layers

    # Every parameter is overwritten by the strict load below; skip the reference's
    # scipy truncated-normal initialisation, which takes minutes for 577M parameters.
    fast = mock.Mock(std=lambda **_: 1.0, rvs=lambda size, **_: np.zeros(size))
    with mock.patch.object(layers, "truncnorm", fast):
        model = Mol(inference_configs())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def convert_checkpoint(checkpoint_path: Path) -> Mol1:
    """Load a Mol-1 release checkpoint (e.g. ``mol-1-base.pt``) into JAX."""
    with torch.no_grad():
        return from_torch(load_reference_model(checkpoint_path))
