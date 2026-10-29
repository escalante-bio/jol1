"""Token-level input embedding, relative position encoding and the distogram head."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jol1.backend import AbstractFromTorch, register_from_torch
from jol1.nn import Linear
from jol1.transformer import AtomAttentionEncoder, AtomFeatures


class TokenFeatures(eqx.Module):
    """Per-token features (see ``jol1.features.Features``)."""

    restype: Float[Array, "T 32"]
    profile: Float[Array, "T 32"]
    deletion_mean: Float[Array, " T"]
    asym_id: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    entity_id: Int[Array, " T"]
    sym_id: Int[Array, " T"]
    token_index: Int[Array, " T"]
    token_bonds: Float[Array, "T T"]
    rep_atom: Int[Array, " T"]  # each token's distogram representative atom


def relative_position_features(
    tokens: TokenFeatures, r_max: int = 32, s_max: int = 2
) -> Float[Array, "T T 139"]:
    """One-hot relative residue / token / chain offsets and the same-entity bit (AF3
    Algorithm 3; the reference's ``generate_relp``)."""

    def pairwise(x):
        return x[:, None], x[None, :]

    same_chain = jnp.equal(*pairwise(tokens.asym_id))
    same_residue = jnp.equal(*pairwise(tokens.residue_index))
    same_entity = jnp.equal(*pairwise(tokens.entity_id))
    d_residue = jnp.clip(jnp.subtract(*pairwise(tokens.residue_index)) + r_max, 0, 2 * r_max)
    d_residue = jnp.where(same_chain, d_residue, 2 * r_max + 1)
    d_token = jnp.clip(jnp.subtract(*pairwise(tokens.token_index)) + r_max, 0, 2 * r_max)
    d_token = jnp.where(same_chain & same_residue, d_token, 2 * r_max + 1)
    d_chain = jnp.clip(jnp.subtract(*pairwise(tokens.sym_id)) + s_max, 0, 2 * s_max)
    d_chain = jnp.where(same_entity, d_chain, 2 * s_max + 1)
    return jnp.concatenate(
        [
            jax.nn.one_hot(d_residue, 2 * r_max + 2),
            jax.nn.one_hot(d_token, 2 * r_max + 2),
            same_entity[..., None].astype(jnp.float32),
            jax.nn.one_hot(d_chain, 2 * s_max + 2),
        ],
        axis=-1,
    )


@register_from_torch("mol.model.modules.embedders.RelativePositionEncoding")
class RelativePositionEncoding(AbstractFromTorch):
    linear_no_bias: Linear
    r_max: int = eqx.field(static=True)
    s_max: int = eqx.field(static=True)

    def __call__(self, relp: Float[Array, "T T R"]) -> Float[Array, "T T Z"]:
        if relp.shape[-1] != 4 * self.r_max + 2 * self.s_max + 7:
            raise ValueError(f"relative position features have {relp.shape[-1]} channels")
        return self.linear_no_bias(relp)


@register_from_torch("mol.model.modules.embedders.InputFeatureEmbedder")
class InputFeatureEmbedder(AbstractFromTorch):
    """AF3 Algorithm 2: atom-attention token embedding + restype, profile, deletion mean."""

    atom_attention_encoder: AtomAttentionEncoder

    def __call__(self, atoms: AtomFeatures, tokens: TokenFeatures) -> Float[Array, "T 449"]:
        encoder = self.atom_attention_encoder
        cond = encoder.conditioning(atoms)
        biases = encoder.atom_transformer.pair_biases(cond.p)
        num_tokens = tokens.restype.shape[0]
        a, _ = encoder(cond.c, cond, biases, atoms.atom_to_token, atoms.atom_mask, num_tokens)
        return jnp.concatenate(
            [a, tokens.restype, tokens.profile, tokens.deletion_mean[:, None]], axis=-1
        )


@register_from_torch("mol.model.modules.embedders.FourierEmbedding")
class FourierEmbedding(AbstractFromTorch):
    """Fixed random Fourier features ``cos(2π (t w + b))`` (AF3 Algorithm 22)."""

    w: Float[Array, " C"]
    b: Float[Array, " C"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " C"]:
        return jnp.cos(2 * math.pi * (t * self.w + self.b))


@register_from_torch("mol.model.modules.head.DistogramHead")
class DistogramHead(AbstractFromTorch):
    linear: Linear

    def __call__(self, z: Float[Array, "T T Z"]) -> Float[Array, "T T Bins"]:
        logits = self.linear(z)
        return logits + jnp.swapaxes(logits, 0, 1)
