"""Diffusion module (AF3 Algorithm 20) split into a step-invariant context and a denoiser.

Every quantity that does not depend on the noisy coordinates or the noise level -- pair
conditioning, atom conditioning, all attention pair biases -- is computed once per
system by :meth:`DiffusionModule.context`; :meth:`DiffusionModule.denoise` then runs one
EDM-preconditioned denoising step. This is the reference's ``full`` diffusion cache,
which it shows to be exact.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jol1.backend import AbstractFromTorch, register_from_torch
from jol1.embedders import FourierEmbedding, RelativePositionEncoding
from jol1.nn import LayerNorm, Linear, Transition
from jol1.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    AtomConditioning,
    AtomFeatures,
    DiffusionTransformer,
)


@register_from_torch("mol.model.modules.diffusion.DiffusionConditioning")
class DiffusionConditioning(AbstractFromTorch):
    """AF3 Algorithm 21, split into the noise-independent parts (:meth:`pair`,
    :meth:`single_base`) and the noise embedding (:meth:`single`)."""

    relpe: RelativePositionEncoding
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    transition_z1: Transition
    transition_z2: Transition
    layernorm_s: LayerNorm
    linear_no_bias_s: Linear
    fourier_embedding: FourierEmbedding
    layernorm_n: LayerNorm
    linear_no_bias_n: Linear
    transition_s1: Transition
    transition_s2: Transition
    sigma_data: float = eqx.field(static=True)

    def pair(self, relp: Float[Array, "T T R"], z_trunk: Float[Array, "T T Z"]):
        z = jnp.concatenate([z_trunk, self.relpe(relp)], axis=-1)
        z = self.linear_no_bias_z(self.layernorm_z(z))
        z = z + self.transition_z1(z)
        return z + self.transition_z2(z)

    def single_base(self, s_trunk: Float[Array, "T S"], s_inputs: Float[Array, "T Ds"]):
        return self.linear_no_bias_s(self.layernorm_s(jnp.concatenate([s_trunk, s_inputs], -1)))

    def single(self, base: Float[Array, "T S"], sigma: Float[Array, ""]) -> Float[Array, "T S"]:
        noise = self.fourier_embedding(jnp.log(sigma / self.sigma_data) / 4)
        s = base + self.linear_no_bias_n(self.layernorm_n(noise))
        s = s + self.transition_s1(s)
        return s + self.transition_s2(s)


class DiffusionContext(eqx.Module):
    """Everything the denoiser needs that is fixed for a system (and a trunk output)."""

    single_base: Float[Array, "T S"]
    atom: AtomConditioning
    token_biases: Float[Array, "blocks heads T T"]
    encoder_biases: Float[Array, "blocks NW heads W H"]
    decoder_biases: Float[Array, "blocks NW heads W H"]
    atom_to_token: Int[Array, " N"]
    atom_mask: Bool[Array, " N"]


@register_from_torch("mol.model.modules.diffusion.DiffusionModule")
class DiffusionModule(AbstractFromTorch):
    diffusion_conditioning: DiffusionConditioning
    atom_attention_encoder: AtomAttentionEncoder
    layernorm_s: LayerNorm
    linear_no_bias_s: Linear
    diffusion_transformer: DiffusionTransformer
    layernorm_a: LayerNorm
    atom_attention_decoder: AtomAttentionDecoder
    sigma_data: float = eqx.field(static=True)
    # Parameter-free LayerNorm of the reference's fused pair-bias path (same math).
    torch_ignore = frozenset({"normalize"})

    def context(
        self,
        atoms: AtomFeatures,
        relp: Float[Array, "T T R"],
        s_inputs: Float[Array, "T Ds"],
        s_trunk: Float[Array, "T S"],
        z_trunk: Float[Array, "T T Z"],
    ) -> DiffusionContext:
        pair = self.diffusion_conditioning.pair(relp, z_trunk)
        atom = self.atom_attention_encoder.conditioning(atoms, s_trunk, pair)
        return DiffusionContext(
            single_base=self.diffusion_conditioning.single_base(s_trunk, s_inputs),
            atom=atom,
            token_biases=self.diffusion_transformer.pair_biases(pair),
            encoder_biases=self.atom_attention_encoder.atom_transformer.pair_biases(atom.p),
            decoder_biases=self.atom_attention_decoder.atom_transformer.pair_biases(atom.p),
            atom_to_token=atoms.atom_to_token,
            atom_mask=atoms.atom_mask,
        )

    def denoise(
        self, context: DiffusionContext, x: Float[Array, "N 3"], sigma: Float[Array, ""]
    ) -> Float[Array, "N 3"]:
        """Denoised coordinates (Å) from noisy coordinates ``x`` at noise level ``sigma``:
        ``c_skip x + c_out F(c_in x, c_noise)`` (EDM; ``c_noise`` is applied in
        :meth:`DiffusionConditioning.single`)."""
        ctx = context
        num_tokens = ctx.single_base.shape[0]
        s = self.diffusion_conditioning.single(ctx.single_base, sigma)

        r = x / jnp.sqrt(self.sigma_data**2 + sigma**2)
        q = ctx.atom.c + self.atom_attention_encoder.linear_no_bias_r(r)
        a, q_skip = self.atom_attention_encoder(
            q, ctx.atom, ctx.encoder_biases, ctx.atom_to_token, ctx.atom_mask, num_tokens
        )
        a = a + self.linear_no_bias_s(self.layernorm_s(s))
        a = self.layernorm_a(self.diffusion_transformer(a, s, ctx.token_biases))
        update = self.atom_attention_decoder(
            a, q_skip, ctx.atom.c, ctx.decoder_biases, ctx.atom_to_token, ctx.atom_mask
        )

        ratio = sigma / self.sigma_data
        return x / (1 + ratio**2) + sigma / jnp.sqrt(1 + ratio**2) * update
