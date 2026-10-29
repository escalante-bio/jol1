"""Mol-1: input embedding, recycled trunk, diffusion denoiser and confidence head.

Each stage is a pure function of the model and its inputs:

1. :meth:`Mol1.embed` -- input embedding (once per system).
2. :meth:`Mol1.trunk` -- ``num_cycles`` passes of templates + MSA module + Pairformer
   stack (or call :meth:`Mol1.cycle` yourself, e.g. to warm-start from a previous state).
3. :meth:`Mol1.diffusion_context` -- step-invariant diffusion conditioning (once per
   trunk output); :meth:`Mol1.denoise` -- one EDM denoising step. Sample with an
   AF3-style sampler using :meth:`Mol1.denoise` and :attr:`Mol1.sampler_settings`.
4. :meth:`Mol1.confidence` -- confidence logits for a sampled structure.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jol1.backend import from_torch, register_from_torch
from jol1.confidence import ConfidenceHead, ConfidenceOutput
from jol1.diffusion import DiffusionContext, DiffusionModule
from jol1.embedders import (
    DistogramHead,
    InputFeatureEmbedder,
    RelativePositionEncoding,
    relative_position_features,
)
from jol1.features import Features
from jol1.nn import LayerNorm, Linear
from jol1.pairformer import MSAModule, PairformerStack, TemplateEmbedder


class InputEmbedding(eqx.Module):
    s_inputs: Float[Array, "T 449"]
    s_init: Float[Array, "T S"]
    z_init: Float[Array, "T T Z"]


class TrunkState(eqx.Module):
    s: Float[Array, "T S"]
    z: Float[Array, "T T Z"]


@dataclasses.dataclass(frozen=True)
class SamplerSettings:
    """The reference's inference noise schedule and Algorithm-18 parameters (from the
    checkpoint config), for building a sampler."""

    sigma_data: float
    s_max: float
    s_min: float
    rho: float
    gamma0: float
    gamma_min: float
    noise_scale_lambda: float
    step_scale_eta: float


def sample_msa_rows(key: PRNGKeyArray, num_rows: int, max_rows: int) -> Bool[Array, " S"]:
    """Row mask of a random MSA subset as the reference draws it at every cycle: a size
    uniform in ``[1, num_rows]`` (capped at ``max_rows``), rows uniformly without
    replacement."""
    k_size, k_perm = jax.random.split(key)
    size = jnp.minimum(jax.random.randint(k_size, (), 1, num_rows + 1), max_rows)
    rank = (
        jnp.zeros(num_rows, jnp.int32)
        .at[jax.random.permutation(k_perm, num_rows)]
        .set(jnp.arange(num_rows))
    )
    return rank < size


@register_from_torch("mol.model.mol.Mol")
class Mol1(eqx.Module):
    input_embedder: InputFeatureEmbedder
    relative_position_encoding: RelativePositionEncoding
    template_embedder: TemplateEmbedder
    msa_module: MSAModule
    pairformer_stack: PairformerStack
    diffusion_module: DiffusionModule
    distogram_head: DistogramHead
    confidence_head: ConfidenceHead
    linear_no_bias_sinit: Linear
    linear_no_bias_zinit1: Linear
    linear_no_bias_zinit2: Linear
    linear_no_bias_token_bond: Linear
    linear_no_bias_z_cycle: Linear
    linear_no_bias_s: Linear
    layernorm_z_cycle: LayerNorm
    layernorm_s: LayerNorm
    sampler_settings: SamplerSettings = eqx.field(static=True)
    msa_max_rows: int = eqx.field(static=True)

    def embed(self, features: Features) -> InputEmbedding:
        """AF3 Algorithm 1 lines 1-5."""
        tokens = features.tokens
        s_inputs = self.input_embedder(features.atoms, tokens)
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[:, None]
            + self.linear_no_bias_zinit2(s_init)[None, :]
            + self.relative_position_encoding(relative_position_features(tokens))
            + self.linear_no_bias_token_bond(tokens.token_bonds[..., None])
        )
        return InputEmbedding(s_inputs=s_inputs, s_init=s_init, z_init=z_init)

    def cycle(
        self,
        features: Features,
        inputs: InputEmbedding,
        state: TrunkState,
        msa_row_mask: Bool[Array, " S"],
    ) -> TrunkState:
        """One trunk pass (AF3 Algorithm 1 lines 8-13) using the MSA rows in ``msa_row_mask``."""
        z = inputs.z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(state.z))
        z = z + self.template_embedder(features.templates, z, features.tokens.asym_id)
        z = self.msa_module(features.msa, z, inputs.s_inputs, msa_row_mask)
        s = inputs.s_init + self.linear_no_bias_s(self.layernorm_s(state.s))
        s, z = self.pairformer_stack(s, z)
        return TrunkState(s=s, z=z)

    def trunk(
        self,
        features: Features,
        inputs: InputEmbedding,
        *,
        num_cycles: int,
        key: PRNGKeyArray | None = None,
        msa_rows: int | None = None,
    ) -> TrunkState:
        """``num_cycles`` trunk passes from a zero state (the reference default is 20).

        ``msa_rows`` keeps only the leading MSA rows (the query first); ``None`` keeps
        all. With ``key``, every pass uses a random subset of them, as the reference does
        at inference; without, every pass uses all of them, so the trunk is deterministic
        (e.g. for design gradients). Use :meth:`cycle` to choose the rows yourself. As in
        the reference, only the last pass is differentiated."""
        if num_cycles < 1:
            raise ValueError(f"num_cycles must be at least 1, got {num_cycles}")
        features = eqx.tree_at(
            lambda f: f.msa, features, jax.tree.map(lambda x: x[:msa_rows], features.msa)
        )
        num_rows = features.msa.msa.shape[0]
        if key is None:
            masks = jnp.ones((num_cycles, num_rows), bool)
        else:
            keys = jax.random.split(key, num_cycles)
            masks = jax.vmap(lambda k: sample_msa_rows(k, num_rows, self.msa_max_rows))(keys)

        state = TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))
        state = jax.lax.fori_loop(
            0,
            num_cycles - 1,
            lambda i, st: jax.lax.stop_gradient(self.cycle(features, inputs, st, masks[i])),
            state,
        )
        return self.cycle(features, inputs, jax.lax.stop_gradient(state), masks[-1])

    def distogram_logits(self, state: TrunkState) -> Float[Array, "T T 64"]:
        return self.distogram_head(state.z)

    def diffusion_context(
        self, features: Features, inputs: InputEmbedding, state: TrunkState
    ) -> DiffusionContext:
        relp = relative_position_features(features.tokens)
        return self.diffusion_module.context(
            features.atoms, relp, inputs.s_inputs, state.s, state.z
        )

    def denoise(
        self, context: DiffusionContext, x: Float[Array, "N 3"], sigma: Float[Array, ""]
    ) -> Float[Array, "N 3"]:
        """EDM x0-prediction (Å) for noisy coordinates ``x`` at noise level ``sigma``."""
        return self.diffusion_module.denoise(context, x, sigma)

    def confidence(
        self,
        features: Features,
        inputs: InputEmbedding,
        state: TrunkState,
        coords: Float[Array, "N 3"],
    ) -> ConfidenceOutput:
        return self.confidence_head(
            inputs.s_inputs,
            state.s,
            state.z,
            coords,
            features.tokens.rep_atom,
            features.atoms.atom_to_token,
            features.atoms.atom_to_tokatom,
        )

    @classmethod
    def from_torch(cls, m):
        names = [
            "input_embedder", "relative_position_encoding", "template_embedder", "msa_module",
            "pairformer_stack", "diffusion_module", "distogram_head", "confidence_head",
            "linear_no_bias_sinit", "linear_no_bias_zinit1", "linear_no_bias_zinit2",
            "linear_no_bias_token_bond", "linear_no_bias_z_cycle", "linear_no_bias_s",
            "layernorm_z_cycle", "layernorm_s",
        ]  # fmt: skip
        if unmatched := {n for n, _ in m.named_children()} - set(names):
            raise ValueError(f"Mol1: unmatched torch children {unmatched}")
        noise, sampler = m.inference_noise_scheduler, m.configs.sample_diffusion
        settings = SamplerSettings(
            sigma_data=float(noise.sigma_data),
            s_max=float(noise.s_max),
            s_min=float(noise.s_min),
            rho=float(noise.rho),
            gamma0=float(sampler.gamma0),
            gamma_min=float(sampler.gamma_min),
            noise_scale_lambda=float(sampler.noise_scale_lambda),
            step_scale_eta=float(sampler.step_scale_eta),
        )
        if settings.sigma_data != m.diffusion_module.sigma_data:
            raise ValueError("noise schedule and diffusion module disagree on sigma_data")
        return cls(
            **{n: from_torch(getattr(m, n)) for n in names},
            sampler_settings=settings,
            msa_max_rows=int(m.msa_module.msa_configs["test_cutoff"]),
        )
