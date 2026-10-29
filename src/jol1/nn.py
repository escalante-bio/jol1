"""Leaf layers mirroring the torch modules Mol-1 is built from."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jol1.backend import AbstractFromTorch, from_torch, register_from_torch


@register_from_torch("torch.nn.Linear")
class Linear(AbstractFromTorch):
    """``nn.Linear`` and its Mol-1 / OpenFold subclasses (which differ only in init)."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, " Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        y = jnp.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias


@register_from_torch("mol.model.triangular.layers.OpenFoldLayerNorm")
@register_from_torch("mol.model.layer_norm.layer_norm.FusedLayerNorm")
class LayerNorm(AbstractFromTorch):
    """Last-axis LayerNorm; the scale and offset are optional (``create_scale/offset``)."""

    weight: Float[Array, " D"] | None
    bias: Float[Array, " D"] | None
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = jax.nn.standardize(x, epsilon=self.eps, algorithm="stable")
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


@register_from_torch("torch.nn.ReLU")
class ReLU(AbstractFromTorch):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


@register_from_torch("torch.nn.modules.container.Sequential")
class Sequential(eqx.Module):
    layers: tuple[eqx.Module, ...]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

    @classmethod
    def from_torch(cls, m):
        return cls(layers=tuple(from_torch(child) for child in m))


@register_from_torch("mol.model.modules.primitives.Transition")
class Transition(AbstractFromTorch):
    """SwiGLU MLP with a pre-LayerNorm (AF3 Algorithm 11)."""

    layernorm1: LayerNorm
    linear_no_bias_a: Linear
    linear_no_bias_b: Linear
    linear_no_bias: Linear

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = self.layernorm1(x)
        return self.linear_no_bias(jax.nn.silu(self.linear_no_bias_a(x)) * self.linear_no_bias_b(x))


@register_from_torch("mol.model.modules.primitives.AdaptiveLayerNorm")
class AdaptiveLayerNorm(AbstractFromTorch):
    """``sigmoid(W s) * LN(a) + W' s`` with ``s`` LayerNorm-ed (AF3 Algorithm 26)."""

    layernorm_a: LayerNorm
    layernorm_s: LayerNorm
    linear_s: Linear
    linear_nobias_s: Linear

    def __call__(self, a: Float[Array, "... A"], s: Float[Array, "... S"]) -> Float[Array, "... A"]:
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        return jax.nn.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)


def scan_layers(step: Callable[[Any, Any], Any], carry: Any, layers: eqx.Module) -> Any:
    """``carry = step(layer, carry)`` for each layer of a stacked module (see
    ``backend.stack_modules``): one rematerialised ``lax.scan`` body for all layers."""
    params, static = eqx.partition(layers, eqx.is_array)

    @jax.checkpoint
    def body(carry, params):
        return step(eqx.combine(params, static), carry), None

    return jax.lax.scan(body, carry, params)[0]
