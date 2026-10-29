"""Confidence head (AF3 Algorithm 31): pLDDT, PAE, PDE and resolved logits for one
predicted structure."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jol1.backend import AbstractFromTorch, register_from_torch
from jol1.nn import LayerNorm, Linear
from jol1.pairformer import PairformerStack


class ConfidenceOutput(eqx.Module):
    plddt_logits: Float[Array, "N 50"]
    pae_logits: Float[Array, "T T 64"]
    pde_logits: Float[Array, "T T 64"]
    resolved_logits: Float[Array, "N 2"]


def pairwise_distances(x: Float[Array, "T 3"]) -> Float[Array, "T T"]:
    """Euclidean distance matrix with a finite gradient on the (zero) diagonal."""
    sq = jnp.sum((x[:, None] - x[None, :]) ** 2, -1)
    nonzero = sq > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sq, 1.0)), 0.0)


@register_from_torch("mol.model.modules.confidence.ConfidenceHead")
class ConfidenceHead(AbstractFromTorch):
    linear_no_bias_s1: Linear
    linear_no_bias_s2: Linear
    lower_bins: Float[Array, " B"]
    upper_bins: Float[Array, " B"]
    linear_no_bias_d: Linear
    linear_no_bias_d_wo_onehot: Linear
    pairformer_stack: PairformerStack
    linear_no_bias_pae: Linear
    linear_no_bias_pde: Linear
    plddt_weight: Float[Array, "A S 50"]
    resolved_weight: Float[Array, "A S 2"]
    input_strunk_ln: LayerNorm
    pae_ln: LayerNorm
    pde_ln: LayerNorm
    plddt_ln: LayerNorm
    resolved_ln: LayerNorm

    def __call__(
        self,
        s_inputs: Float[Array, "T Ds"],
        s_trunk: Float[Array, "T S"],
        z_trunk: Float[Array, "T T Z"],
        coords: Float[Array, "N 3"],
        rep_atom: Int[Array, " T"],
        atom_to_token: Int[Array, " N"],
        atom_to_tokatom: Int[Array, " N"],
    ) -> ConfidenceOutput:
        """Unlike the reference, which detaches ``s_inputs``, ``s_trunk`` and ``z_trunk``,
        gradients flow through every input."""
        s = self.input_strunk_ln(jnp.clip(s_trunk, -512, 512))
        z = (
            z_trunk
            + self.linear_no_bias_s1(s_inputs)[None, :]
            + self.linear_no_bias_s2(s_inputs)[:, None]
        )
        d = pairwise_distances(coords[rep_atom])[..., None]
        z = z + self.linear_no_bias_d(
            ((d > self.lower_bins) & (d < self.upper_bins)).astype(z.dtype)
        )
        z = z + self.linear_no_bias_d_wo_onehot(d)

        s, z = self.pairformer_stack(s, z)

        a = self.plddt_ln(s)[atom_to_token]
        b = self.resolved_ln(s)[atom_to_token]
        return ConfidenceOutput(
            plddt_logits=jnp.einsum("nc,ncb->nb", a, self.plddt_weight[atom_to_tokatom]),
            pae_logits=self.linear_no_bias_pae(self.pae_ln(z)),
            pde_logits=self.linear_no_bias_pde(self.pde_ln(z + jnp.swapaxes(z, 0, 1))),
            resolved_logits=jnp.einsum("nc,ncb->nb", b, self.resolved_weight[atom_to_tokatom]),
        )
