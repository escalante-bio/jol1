"""Pair-representation stacks: triangle updates, PairMixer / Pairformer blocks, the MSA
module and the template embedder.

Everything is unbatched (``T`` tokens, ``S`` MSA rows) and, like the reference at
inference, unmasked: Mol-1 features have no token padding. The only mask is an MSA row
mask, used to subsample the MSA without changing shapes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jol1.backend import AbstractFromTorch, from_torch, register_converter, register_from_torch
from jol1.nn import LayerNorm, Linear, Transition, scan_layers
from jol1.transformer import Attention, AttentionPairBias


@register_converter("mol.model.triangular.layers.Attention")
def _openfold_attention(m) -> Attention:
    """OpenFold's gated attention (triangle attention): no query bias, else identical."""
    return Attention(
        linear_q=from_torch(m.linear_q),
        linear_k=from_torch(m.linear_k),
        linear_v=from_torch(m.linear_v),
        linear_o=from_torch(m.linear_o),
        linear_g=from_torch(m.linear_g),
        num_heads=m.no_heads,
    )


@register_from_torch("mol.model.triangular.triangular.TriangleMultiplicationOutgoing")
@register_from_torch("mol.model.triangular.triangular.TriangleMultiplicationIncoming")
class TriangleMultiplication(AbstractFromTorch):
    """AF3 Algorithms 12/13 (all reference kernels share these weights and this math)."""

    layer_norm_in: LayerNorm
    linear_a_p: Linear
    linear_a_g: Linear
    linear_b_p: Linear
    linear_b_g: Linear
    layer_norm_out: LayerNorm
    linear_z: Linear
    linear_g: Linear
    outgoing: bool = eqx.field(static=True)
    torch_ignore = frozenset({"sigmoid"})

    def __call__(self, z: Float[Array, "T T Z"]) -> Float[Array, "T T Z"]:
        z = self.layer_norm_in(z)
        a = jax.nn.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = jax.nn.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        spec = "ikc,jkc->ijc" if self.outgoing else "kic,kjc->ijc"
        x = self.linear_z(self.layer_norm_out(jnp.einsum(spec, a, b)))
        return x * jax.nn.sigmoid(self.linear_g(z))

    @classmethod
    def from_torch(cls, m):
        return super().from_torch(m, outgoing=m._outgoing)


@register_from_torch("mol.model.triangular.triangular.TriangleAttention")
class TriangleAttention(AbstractFromTorch):
    """Starting-node triangle attention (AF3 Algorithm 14): row ``i`` attends over ``j``
    with a per-head bias from the pair itself. (Ending-node attention is this module
    applied to the transposed pair, as in the reference's ``PairformerBlock``.)"""

    layer_norm: LayerNorm
    linear: Linear
    mha: Attention

    def __call__(self, z: Float[Array, "T T Z"]) -> Float[Array, "T T Z"]:
        z = self.layer_norm(z)
        bias = jnp.moveaxis(self.linear(z), -1, 0)  # [heads, J, K], shared by all rows
        return self.mha(z, z, bias[None])

    @classmethod
    def from_torch(cls, m):
        if not m.starting:
            raise ValueError("Mol-1 only uses starting-node TriangleAttention")
        return super().from_torch(m)


@register_from_torch("mol.model.triangular.layers.OuterProductMean")
class OuterProductMean(AbstractFromTorch):
    """AF3 Algorithm 9 over the MSA rows selected by ``row_mask``."""

    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: float = eqx.field(static=True)

    def __call__(
        self, m: Float[Array, "S T C"], row_mask: Bool[Array, " S"]
    ) -> Float[Array, "T T Z"]:
        m = self.layer_norm(m)
        w = row_mask[:, None, None].astype(m.dtype)
        a, b = self.linear_1(m) * w, self.linear_2(m) * w
        outer = jnp.einsum("sic,sje->ijce", a, b)
        outer = self.linear_out(outer.reshape(*outer.shape[:2], -1))
        return outer / (jnp.sum(w) + self.eps)


@register_from_torch("mol.model.modules.pairformer.MSAPairWeightedAveraging")
class MSAPairWeightedAveraging(AbstractFromTorch):
    """AF3 Algorithm 10: each MSA row averages its values with pair-derived weights."""

    layernorm_m: LayerNorm
    linear_no_bias_mv: Linear
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    linear_no_bias_mg: Linear
    linear_no_bias_out: Linear
    n_heads: int = eqx.field(static=True)
    torch_ignore = frozenset({"softmax_w"})

    def __call__(self, m: Float[Array, "S T C"], z: Float[Array, "T T Z"]) -> Float[Array, "S T C"]:
        m = self.layernorm_m(m)
        heads = (*m.shape[:2], self.n_heads, -1)
        v = self.linear_no_bias_mv(m).reshape(heads)
        g = jax.nn.sigmoid(self.linear_no_bias_mg(m)).reshape(heads)
        w = jax.nn.softmax(self.linear_no_bias_z(self.layernorm_z(z)), axis=1)  # over j
        o = g * jnp.einsum("ijh,sjhc->sihc", w, v)
        return self.linear_no_bias_out(o.reshape(*m.shape[:2], -1))


@register_from_torch("mol.model.modules.pairformer.MSAStack")
class MSAStack(AbstractFromTorch):
    msa_pair_weighted_averaging: MSAPairWeightedAveraging
    transition_m: Transition
    torch_ignore = frozenset({"dropout_row"})

    def __call__(self, m: Float[Array, "S T C"], z: Float[Array, "T T Z"]) -> Float[Array, "S T C"]:
        m = m + self.msa_pair_weighted_averaging(m, z)
        return m + self.transition_m(m)


@register_from_torch("mol.model.modules.pairformer.PairformerBlock")
class PairformerBlock(AbstractFromTorch):
    """AF3 Algorithm 17 block. Without a single track (``c_s = 0``: MSA / template stacks)
    the attention fields are ``None``."""

    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    pair_transition: Transition
    attention_pair_bias: AttentionPairBias | None = None
    single_transition: Transition | None = None
    torch_ignore = frozenset({"dropout_row"})

    def pair_update(self, z: Float[Array, "T T Z"]) -> Float[Array, "T T Z"]:
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_att_start(z)
        z = z + jnp.swapaxes(self.tri_att_end(jnp.swapaxes(z, 0, 1)), 0, 1)
        return z + self.pair_transition(z)

    def __call__(self, s, z):
        return single_update(self, s, self.pair_update(z))


@register_from_torch("mol.model.modules.pairformer.PairMixerBlock")
class PairMixerBlock(AbstractFromTorch):
    """A Pairformer block without triangle attention."""

    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    pair_transition: Transition
    attention_pair_bias: AttentionPairBias | None = None
    single_transition: Transition | None = None
    torch_ignore = frozenset({"dropout_row"})

    def __call__(self, s, z):
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.pair_transition(z)
        return single_update(self, s, z)


def single_update(block: PairformerBlock | PairMixerBlock, s, z):
    """Single-track update shared by both block types (a no-op without a single track)."""
    if block.attention_pair_bias is None:
        return s, z
    apb = block.attention_pair_bias
    s = s + apb(s, None, apb.pair_bias(z))
    return s + block.single_transition(s), z


@register_from_torch("mol.model.modules.pairformer.PairformerStack")
class PairformerStack(eqx.Module):
    """Leading PairMixer blocks then Pairformer blocks, each group at its own pair width.

    Linear adapters project the pair representation into the Pairformer width and back
    to the resident width when they differ (Mol-1's trunk: 8 PairMixer blocks at 448,
    40 Pairformer blocks at 320)."""

    pairmixer_blocks: PairMixerBlock | None  # stacked
    pairformer_adapter: Linear | None
    pairformer_blocks: PairformerBlock | None  # stacked
    output_adapter: Linear | None

    def __call__(
        self, s: Float[Array, "T S"] | None, z: Float[Array, "T T Z"]
    ) -> tuple[Float[Array, "T S"] | None, Float[Array, "T T Z"]]:
        def step(block, carry):
            return block(*carry)

        if self.pairmixer_blocks is not None:
            s, z = scan_layers(step, (s, z), self.pairmixer_blocks)
        if self.pairformer_adapter is not None:
            z = self.pairformer_adapter(z)
        if self.pairformer_blocks is not None:
            s, z = scan_layers(step, (s, z), self.pairformer_blocks)
        if self.output_adapter is not None:
            z = self.output_adapter(z)
        return s, z

    @classmethod
    def from_torch(cls, m):
        from jol1.backend import stack_modules

        n_mix = m.n_pairmixer_blocks
        blocks = [from_torch(b) for b in m.blocks]
        expected = {f"before_block_{n_mix}", "after_blocks"} if m.c_z_pairformer != m.c_z else set()
        if n_mix and m.c_z_pairmixer != m.c_z:
            raise ValueError("PairMixer blocks must run at the resident pair width")
        if set(m.adapters.keys()) != expected:
            raise ValueError(f"unexpected Pairformer adapters {list(m.adapters.keys())}")

        def adapter(name):
            return from_torch(m.adapters[name].pair_adapter) if name in expected else None

        return cls(
            pairmixer_blocks=stack_modules(blocks[:n_mix]) if n_mix else None,
            pairformer_adapter=adapter(f"before_block_{n_mix}"),
            pairformer_blocks=stack_modules(blocks[n_mix:]) if n_mix < len(blocks) else None,
            output_adapter=adapter("after_blocks"),
        )


@register_from_torch("mol.model.modules.pairformer.MSABlock")
class MSABlock(AbstractFromTorch):
    """AF3 Algorithm 8 block. The last block has no MSA stack (its MSA output is unused)."""

    outer_product_mean_msa: OuterProductMean
    pair_stack: PairformerBlock
    msa_stack: MSAStack | None = None

    def __call__(self, m, z, row_mask):
        z = z + self.outer_product_mean_msa(m, row_mask)
        if self.msa_stack is not None:
            m = self.msa_stack(m, z)
        return m, self.pair_stack.pair_update(z)


class MSAFeatures(eqx.Module):
    """MSA features (see ``jol1.features.Features``). The MSA is one-hot (or soft) over the
    32 residue types, so a row can carry a sequence profile."""

    msa: Float[Array, "S T 32"]
    has_deletion: Float[Array, "S T"]
    deletion_value: Float[Array, "S T"]


@register_from_torch("mol.model.modules.pairformer.MSAModule")
class MSAModule(eqx.Module):
    """AF3 Algorithm 8. ``row_mask`` selects the MSA rows used (the reference subsamples
    rows at random; rows never interact except through the outer-product mean, so
    masking rows out is exactly equivalent to dropping them)."""

    linear_no_bias_m: Linear
    linear_no_bias_s: Linear
    blocks: MSABlock  # stacked, all but the last
    last_block: MSABlock

    def __call__(
        self,
        msa: MSAFeatures,
        z: Float[Array, "T T Z"],
        s_inputs: Float[Array, "T Ds"],
        row_mask: Bool[Array, " S"],
    ) -> Float[Array, "T T Z"]:
        m = jnp.concatenate(
            [
                msa.msa,
                msa.has_deletion[..., None],
                msa.deletion_value[..., None],
            ],
            axis=-1,
        )
        m = self.linear_no_bias_m(m) + self.linear_no_bias_s(s_inputs)
        m, z = scan_layers(lambda block, mz: block(*mz, row_mask), (m, z), self.blocks)
        return eqx.filter_checkpoint(self.last_block)(m, z, row_mask)[1]

    @classmethod
    def from_torch(cls, m):
        from jol1.backend import stack_modules

        blocks = [from_torch(b) for b in m.blocks]
        return cls(
            linear_no_bias_m=from_torch(m.linear_no_bias_m),
            linear_no_bias_s=from_torch(m.linear_no_bias_s),
            blocks=stack_modules(blocks[:-1]),
            last_block=blocks[-1],
        )


class TemplateFeatures(eqx.Module):
    """Template features (see ``jol1.features.Features``); empty templates are all-gap."""

    aatype: Int[Array, "Nt T"]
    distogram: Float[Array, "Nt T T 39"]
    pseudo_beta_mask: Float[Array, "Nt T T"]
    unit_vector: Float[Array, "Nt T T 3"]
    backbone_frame_mask: Float[Array, "Nt T T"]


@register_from_torch("mol.model.modules.pairformer.TemplateEmbedder")
class TemplateEmbedder(AbstractFromTorch):
    """AF3 Algorithm 16: every template (empty ones included, as in the reference) runs
    through a small Pairformer; the results are averaged."""

    linear_no_bias_z: Linear
    layernorm_z: LayerNorm
    linear_no_bias_a: Linear
    pairformer_stack: PairformerStack
    layernorm_v: LayerNorm
    linear_no_bias_u: Linear
    torch_ignore = frozenset({"relu"})

    def __call__(
        self,
        templates: TemplateFeatures,
        z: Float[Array, "T T Z"],
        asym_id: Int[Array, " T"],
    ) -> Float[Array, "T T Z"]:
        T = z.shape[0]
        same_chain = (asym_id[:, None] == asym_id[None, :]).astype(z.dtype)
        z = self.linear_no_bias_z(self.layernorm_z(z))

        def one_template(u, t):
            aatype = jax.nn.one_hot(t.aatype, 32)
            at = jnp.concatenate(
                [
                    t.distogram * same_chain[..., None],
                    (t.pseudo_beta_mask * same_chain)[..., None],
                    jnp.broadcast_to(aatype[None, :], (T, T, 32)),
                    jnp.broadcast_to(aatype[:, None], (T, T, 32)),
                    t.unit_vector * same_chain[..., None],
                    (t.backbone_frame_mask * same_chain)[..., None],
                ],
                axis=-1,
            )
            v = self.pairformer_stack(None, z + self.linear_no_bias_a(at))[1]
            return u + self.layernorm_v(v), None

        num_templates = templates.aatype.shape[0]
        u0 = jnp.zeros((T, T, self.layernorm_v.weight.shape[0]), z.dtype)
        u = jax.lax.scan(one_template, u0, templates)[0] / (1e-7 + num_templates)
        return self.linear_no_bias_u(jax.nn.relu(u))
