"""Attention with pair bias, adaLN transformer blocks and the atom attention encoder/decoder.

Token-level attention is full self-attention with a per-head pair bias. Atom-level
attention is sequence-local: atoms are split into query windows of ``W = 32`` atoms,
each attending to the ``H = 128`` atoms centred on it (the reference's
``rearrange_qk_to_dense_trunk``). Atom-pair tensors live in that windowed layout,
``[NW, W, H, C]`` with ``NW = ceil(N / W)``.

Atoms may be padded (``AtomFeatures.atom_mask``; the reference never pads): padding atoms
are never attended to and do not contribute to token aggregates, so the real atoms'
outputs do not depend on them.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jol1.backend import (
    AbstractFromTorch,
    from_torch,
    register_converter,
    register_from_torch,
    stacked,
)
from jol1.nn import AdaptiveLayerNorm, LayerNorm, Linear, Sequential

W, H = 32, 128  # atom query / key window sizes (n_queries, n_keys in the reference)


def key_window_index(num_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Static gather table for sliding-window atom attention.

    Query window ``k`` covers atoms ``[k*W, (k+1)*W)`` and attends to atoms
    ``k*W - (H-W)/2 + j``, ``j < H``. Returns ``(index [NW, H], in_range [NW, H])``;
    out-of-range slots are clamped to 0 and must be masked.
    """
    num_windows = -(-num_atoms // W)
    index = np.arange(num_windows)[:, None] * W - (H - W) // 2 + np.arange(H)[None]
    in_range = (index >= 0) & (index < num_atoms)
    return np.where(in_range, index, 0), in_range


def window_queries(x: Float[Array, "N ..."]) -> Float[Array, "NW W ..."]:
    """Per-atom values in query-window layout (zero padding past the last atom)."""
    num_windows = -(-x.shape[0] // W)
    pad = [(0, num_windows * W - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, pad).reshape(num_windows, W, *x.shape[1:])


def window_keys(x: Float[Array, "N ..."]) -> Float[Array, "NW H ..."]:
    """Per-atom values in key-window layout (zero outside the atom range)."""
    index, in_range = key_window_index(x.shape[0])
    in_range = in_range.reshape(in_range.shape + (1,) * (x.ndim - 1))
    gathered = x[index]
    return jnp.where(in_range, gathered, jnp.zeros_like(gathered))


def unwindow(x: Float[Array, "NW W ..."], num_atoms: int) -> Float[Array, "N ..."]:
    return x.reshape(-1, *x.shape[2:])[:num_atoms]


def broadcast_token_to_atom(
    x: Float[Array, "T D"], atom_to_token: Int[Array, " N"]
) -> Float[Array, "N D"]:
    return x[atom_to_token]


def mean_atoms_to_tokens(
    x: Float[Array, "N D"],
    atom_to_token: Int[Array, " N"],
    atom_mask: Bool[Array, " N"],
    num_tokens: int,
) -> Float[Array, "T D"]:
    """Per-token mean over the token's real atoms (``torch_scatter`` mean: empty tokens
    give 0)."""
    segment = jnp.where(atom_mask, atom_to_token, num_tokens)  # padding -> dump slot
    total = jax.ops.segment_sum(x, segment, num_segments=num_tokens + 1)[:num_tokens]
    count = jax.ops.segment_sum(jnp.ones_like(segment, x.dtype), segment, num_tokens + 1)
    return total / jnp.maximum(count[:num_tokens], 1)[:, None]


@register_from_torch("mol.model.modules.primitives.Attention")
class Attention(AbstractFromTorch):
    """Gated multi-head attention (q has a bias; k, v, gate and output do not)."""

    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear
    num_heads: int = eqx.field(static=True)
    torch_ignore = frozenset({"sigmoid"})

    def _heads(self, x: Float[Array, "L D"]) -> Float[Array, "L heads C"]:
        return x.reshape(*x.shape[:-1], self.num_heads, -1)

    def _output(self, o: Float[Array, "L heads C"], q_x: Float[Array, "L D"]):
        o = o * self._heads(jax.nn.sigmoid(self.linear_g(q_x)))
        return self.linear_o(o.reshape(*o.shape[:-2], -1))

    def __call__(
        self,
        q_x: Float[Array, "Q D"],
        kv_x: Float[Array, "K D"],
        bias: Float[Array, "heads Q K"],
    ) -> Float[Array, "Q D"]:
        q = self._heads(self.linear_q(q_x))
        k = self._heads(self.linear_k(kv_x))
        v = self._heads(self.linear_v(kv_x))
        return self._output(jax.nn.dot_product_attention(q, k, v, bias=bias), q_x)

    def local(
        self,
        q_x: Float[Array, "N D"],
        kv_x: Float[Array, "N D"],
        bias: Float[Array, "NW heads W H"],
        atom_mask: Bool[Array, " N"],
    ) -> Float[Array, "N D"]:
        """Sliding-window attention: each query window attends to the real atoms of its
        key window."""
        N = q_x.shape[0]
        q = window_queries(self._heads(self.linear_q(q_x)))
        k = window_keys(self._heads(self.linear_k(kv_x)))
        v = window_keys(self._heads(self.linear_v(kv_x)))
        key_mask = window_keys(atom_mask)  # False outside the atom range
        o = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=key_mask[:, None, None, :])
        return self._output(unwindow(o, N), q_x)


@register_from_torch("mol.model.modules.transformer.AttentionPairBias")
class AttentionPairBias(AbstractFromTorch):
    """AF3 Algorithm 24. With conditioning ``s`` (diffusion blocks) the input is adaLN-ed and
    the output gated by ``s``; without (Pairformer) it is plainly LayerNorm-ed. In
    cross-attention mode (atom blocks) keys/values get their own adaLN, applied -- as in
    the reference -- to the already adaLN-ed queries."""

    layernorm_a: LayerNorm | AdaptiveLayerNorm
    attention: Attention
    layernorm_z: LayerNorm
    linear_nobias_z: Linear
    layernorm_kv: AdaptiveLayerNorm | None = None
    linear_a_last: Linear | None = None

    def pair_bias(self, z: Float[Array, "... Q K Z"]) -> Float[Array, "... heads Q K"]:
        return jnp.moveaxis(self.linear_nobias_z(self.layernorm_z(z)), -1, -3)

    def _inputs(self, a, s):
        if self.linear_a_last is None:
            a = self.layernorm_a(a)
        else:
            a = self.layernorm_a(a, s)
        kv = a if self.layernorm_kv is None else self.layernorm_kv(a, s)
        return a, kv

    def _gate(self, a, s):
        return a if self.linear_a_last is None else jax.nn.sigmoid(self.linear_a_last(s)) * a

    def __call__(
        self,
        a: Float[Array, "T D"],
        s: Float[Array, "T S"] | None,
        bias: Float[Array, "heads T T"],
    ) -> Float[Array, "T D"]:
        q, kv = self._inputs(a, s)
        return self._gate(self.attention(q, kv, bias), s)

    def local(
        self,
        a: Float[Array, "N D"],
        s: Float[Array, "N S"],
        bias: Float[Array, "NW heads W H"],
        atom_mask: Bool[Array, " N"],
    ) -> Float[Array, "N D"]:
        q, kv = self._inputs(a, s)
        return self._gate(self.attention.local(q, kv, bias, atom_mask), s)


@register_from_torch("mol.model.modules.transformer.ConditionedTransitionBlock")
class ConditionedTransitionBlock(AbstractFromTorch):
    """AF3 Algorithm 25: adaLN, SwiGLU, output gated by the conditioning."""

    adaln: AdaptiveLayerNorm
    linear_nobias_a1: Linear
    linear_nobias_a2: Linear
    linear_nobias_b: Linear
    linear_s: Linear

    def __call__(self, a: Float[Array, "L D"], s: Float[Array, "L S"]) -> Float[Array, "L D"]:
        a = self.adaln(a, s)
        b = jax.nn.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)
        return jax.nn.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)


@register_from_torch("mol.model.modules.transformer.DiffusionTransformerBlock")
class DiffusionTransformerBlock(AbstractFromTorch):
    attention_pair_bias: AttentionPairBias
    conditioned_transition_block: ConditionedTransitionBlock
    torch_ignore = frozenset({"drop_path"})

    def __call__(self, a, s, bias):
        a = a + self.attention_pair_bias(a, s, bias)
        return a + self.conditioned_transition_block(a, s)

    def local(self, a, s, bias, atom_mask):
        a = a + self.attention_pair_bias.local(a, s, bias, atom_mask)
        return a + self.conditioned_transition_block(a, s)


@register_from_torch("mol.model.modules.transformer.DiffusionTransformer")
class DiffusionTransformer(AbstractFromTorch):
    """AF3 Algorithm 23 (and, via :meth:`local`, the atom transformer of Algorithm 7).

    The pair input only enters through per-block attention biases, which depend on
    nothing that changes between denoising steps: compute them once with
    :meth:`pair_biases`."""

    blocks: DiffusionTransformerBlock = stacked()

    def pair_biases(self, z: Float[Array, "*P Z"]) -> Float[Array, "blocks ..."]:
        """Per-block attention biases: ``[blocks, heads, T, T]`` from token pairs
        ``[T, T, Z]``, or ``[blocks, NW, heads, W, H]`` from windowed atom pairs."""
        return eqx.filter_vmap(lambda b: b.attention_pair_bias.pair_bias(z))(self.blocks)

    def __call__(
        self,
        a: Float[Array, "T D"],
        s: Float[Array, "T S"],
        biases: Float[Array, "blocks heads T T"],
    ) -> Float[Array, "T D"]:
        """Full self-attention over tokens."""
        return self._scan(lambda block, a, bias: block(a, s, bias), a, biases)

    def local(
        self,
        a: Float[Array, "N D"],
        s: Float[Array, "N S"],
        biases: Float[Array, "blocks NW heads W H"],
        atom_mask: Bool[Array, " N"],
    ) -> Float[Array, "N D"]:
        """Sliding-window self-attention over (real) atoms."""
        return self._scan(lambda block, a, bias: block.local(a, s, bias, atom_mask), a, biases)

    def _scan(self, apply, a, biases):
        params, static = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def body(a, xs):
            block_params, bias = xs
            return apply(eqx.combine(block_params, static), a, bias), None

        return jax.lax.scan(body, a, (params, biases))[0]


@register_converter("mol.model.modules.transformer.AtomTransformer")
def _atom_transformer(m) -> DiffusionTransformer:
    """``AtomTransformer`` is a ``DiffusionTransformer`` with fixed atom windows."""
    if (m.n_queries, m.n_keys) != (W, H):
        raise ValueError(f"atom windows must be ({W}, {H}), got ({m.n_queries}, {m.n_keys})")
    return from_torch(m.diffusion_transformer)


class AtomFeatures(eqx.Module):
    """Per-atom reference-conformer features (see ``jol1.features.Features``)."""

    ref_pos: Float[Array, "N 3"]
    ref_charge: Float[Array, " N"]
    ref_mask: Float[Array, " N"]
    ref_element: Float[Array, "N 128"]
    ref_atom_name_chars: Float[Array, "N 256"]
    ref_space_uid: Int[Array, " N"]
    atom_to_token: Int[Array, " N"]
    atom_to_tokatom: Int[Array, " N"]
    atom_mask: Bool[Array, " N"]  # False for padding atoms


class AtomConditioning(eqx.Module):
    """Step-invariant atom-level conditioning: single ``c`` and windowed pair ``p``."""

    c: Float[Array, "N Ca"]
    p: Float[Array, "NW W H Cp"]


@register_from_torch("mol.model.modules.transformer.AtomAttentionEncoder")
class AtomAttentionEncoder(AbstractFromTorch):
    """AF3 Algorithm 5. Without coordinates (input embedder) it embeds the reference
    conformer; with coordinates (diffusion) it adds trunk conditioning and the noisy
    positions. Everything but the positions is step-invariant (:meth:`conditioning`)."""

    linear_no_bias_ref_pos: Linear
    linear_no_bias_ref_charge: Linear
    linear_no_bias_f: Linear
    linear_no_bias_d: Linear
    linear_no_bias_invd: Linear
    linear_no_bias_v: Linear
    linear_no_bias_cl: Linear
    linear_no_bias_cm: Linear
    small_mlp: Sequential
    atom_transformer: DiffusionTransformer
    linear_no_bias_q: Linear
    # Only with coordinates (the diffusion module's encoder).
    layernorm_s: LayerNorm | None = None
    linear_no_bias_s: Linear | None = None
    layernorm_z: LayerNorm | None = None
    linear_no_bias_z: Linear | None = None
    linear_no_bias_r: Linear | None = None

    def conditioning(
        self,
        atoms: AtomFeatures,
        s_trunk: Float[Array, "T S"] | None = None,
        z_pair: Float[Array, "T T Z"] | None = None,
    ) -> AtomConditioning:
        """Algorithm 5 lines 1-14 (everything that does not depend on the positions).
        ``s_trunk`` and ``z_pair`` are given exactly when the encoder has coordinates."""
        c = (
            self.linear_no_bias_ref_pos(atoms.ref_pos)
            + self.linear_no_bias_ref_charge(jnp.arcsinh(atoms.ref_charge)[:, None])
            + self.linear_no_bias_f(
                jnp.concatenate(
                    [atoms.ref_mask[:, None], atoms.ref_element, atoms.ref_atom_name_chars], -1
                )
            )
        ) * atoms.ref_mask[:, None]

        _, in_range = key_window_index(atoms.ref_pos.shape[0])
        valid = (
            window_queries(jnp.ones(atoms.ref_pos.shape[0], bool))[:, :, None] & in_range[:, None]
        )
        d = window_queries(atoms.ref_pos)[:, :, None] - window_keys(atoms.ref_pos)[:, None]
        uid_q = window_queries(atoms.ref_space_uid)[:, :, None]
        v = (uid_q == window_keys(atoms.ref_space_uid)[:, None])[..., None].astype(d.dtype)
        p = self.linear_no_bias_d(d) * v * valid[..., None]
        p = p + self.linear_no_bias_invd(1 / (1 + jnp.sum(d**2, -1, keepdims=True))) * v
        p = p + self.linear_no_bias_v(v)

        if s_trunk is not None:
            tok = atoms.atom_to_token
            c = c + broadcast_token_to_atom(self.linear_no_bias_s(self.layernorm_s(s_trunk)), tok)
            z = self.linear_no_bias_z(self.layernorm_z(z_pair))
            p = p + z[window_queries(tok)[:, :, None], window_keys(tok)[:, None]]

        p = (
            p
            + self.linear_no_bias_cl(jax.nn.relu(window_queries(c)))[:, :, None]
            + self.linear_no_bias_cm(jax.nn.relu(window_keys(c)))[:, None]
        )
        return AtomConditioning(c=c, p=p + self.small_mlp(p))

    def __call__(
        self,
        q: Float[Array, "N Ca"],
        cond: AtomConditioning,
        biases: Float[Array, "blocks NW heads W H"],
        atom_to_token: Int[Array, " N"],
        atom_mask: Bool[Array, " N"],
        num_tokens: int,
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        """Atom transformer and mean-aggregation to tokens. Returns ``(a, q)``."""
        q = self.atom_transformer.local(q, cond.c, biases, atom_mask)
        a = jax.nn.relu(self.linear_no_bias_q(q))
        return mean_atoms_to_tokens(a, atom_to_token, atom_mask, num_tokens), q


@register_from_torch("mol.model.modules.transformer.AtomAttentionDecoder")
class AtomAttentionDecoder(AbstractFromTorch):
    """AF3 Algorithm 6: broadcast tokens to atoms, atom transformer, project to positions."""

    linear_no_bias_a: Linear
    layernorm_q: LayerNorm
    linear_no_bias_out: Linear
    atom_transformer: DiffusionTransformer

    def __call__(
        self,
        a: Float[Array, "T Ct"],
        q_skip: Float[Array, "N Ca"],
        c_skip: Float[Array, "N Ca"],
        biases: Float[Array, "blocks NW heads W H"],
        atom_to_token: Int[Array, " N"],
        atom_mask: Bool[Array, " N"],
    ) -> Float[Array, "N 3"]:
        q = broadcast_token_to_atom(self.linear_no_bias_a(a), atom_to_token) + q_skip
        q = self.atom_transformer.local(q, c_skip, biases, atom_mask)
        return self.linear_no_bias_out(self.layernorm_q(q))
