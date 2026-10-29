"""Parity vs the torch reference with the released checkpoint, on a real system.

Needs the ``reference`` extra, the checkpoint at ``mol_data/checkpoint/mol-1-base.pt``
and captures from ``tests/capture.py``. Run with ``pytest -m reference``.

Every captured module call is replayed through the JAX module converted from that
torch submodule; then the stages (input embedding, recycled trunk, diffusion context
and denoising, confidence) run end to end.
"""

from __future__ import annotations

import pickle
from functools import cache
from pathlib import Path
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.reference

import torch  # noqa: E402  (fails loudly without the `reference` extra)

import jol1.confidence  # noqa: E402
from jol1 import Features, InputEmbedding, TrunkState  # noqa: E402
from jol1.convert import from_torch, load_reference_model  # noqa: E402
from jol1.embedders import relative_position_features  # noqa: E402
from jol1.pairformer import MSAFeatures, TemplateFeatures  # noqa: E402
from jol1.transformer import AtomFeatures  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
torch.set_float32_matmul_precision("highest")

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
CHECKPOINT = ROOT.parent / "mol_data/checkpoint/mol-1-base.pt"
NUM_CYCLES = 3  # must match tests/capture.py


def rel_err(ours, ref) -> float:
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    return float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-12))


def check(ours, ref, tol):
    err = rel_err(ours, ref)
    print(f"max rel err {err:.1e}", end="  ")
    assert err < tol, f"max rel err {err:.2e} >= {tol:.0e}"


@cache
def capture(name: str) -> dict:
    with open(CAPTURES / "modules" / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


@cache
def run() -> dict:
    with open(CAPTURES / "run.pkl", "rb") as f:
        return pickle.load(f)


@cache
def reference() -> torch.nn.Module:
    return load_reference_model(CHECKPOINT)


@cache
def model():
    with torch.no_grad():
        return from_torch(reference())


@cache
def features() -> Features:
    return Features.from_reference_dict(run()["features"])


def squeeze(x, ndim):
    """Drop the reference's leading singleton sample/batch axes."""
    x = np.asarray(x)
    while x.ndim > ndim:
        assert x.shape[0] == 1, x.shape
        x = x[0]
    return jnp.asarray(x)


# ---------------------------------------------------------------- per-module replay
# Each adapter maps a captured torch call (positional args, kwargs) to the JAX module
# and returns ``(ours, reference)`` pairs.


def no_padding(x) -> jax.Array:
    """All-real atom mask (the reference never pads atoms)."""
    return jnp.ones(x.shape[0], bool)


def get(rec, index, name):
    return rec["kwargs"][name] if name in rec["kwargs"] else rec["args"][index]


def _unary(m, rec):
    return [(m(jnp.asarray(rec["args"][0])), rec["output"])]


def _adaln(m, rec):
    return [(m(get(rec, 0, "a"), get(rec, 1, "s")), rec["output"])]


def _tri_attention(m, rec):
    return [(m(jnp.asarray(get(rec, 0, "x"))), rec["output"])]


def _openfold_attention(m, rec):
    bias = sum(np.asarray(b) for b in rec["kwargs"]["biases"])
    return [(m(rec["kwargs"]["q_x"], rec["kwargs"]["kv_x"], bias), rec["output"])]


def _attention(m, rec):
    kw = rec["kwargs"]
    q_x, kv_x = squeeze(kw["q_x"], 2), squeeze(kw["kv_x"], 2)
    if kw.get("n_queries"):
        bias = np.moveaxis(squeeze(kw["trunked_attn_bias"], 4), 0, 1)  # [NW, heads, W, H]
        return [(m.local(q_x, kv_x, bias, no_padding(q_x)), squeeze(rec["output"], 2))]
    return [(m(q_x, kv_x, squeeze(kw["attn_bias"], 3)), squeeze(rec["output"], 2))]


def _apb_inputs(m, kw):
    a = squeeze(kw["a"], 2)
    s = None if kw["s"] is None else squeeze(kw["s"], 2)
    local = bool(kw.get("n_queries"))
    bias = m.pair_bias(squeeze(kw["z"], 4 if local else 3))
    if kw.get("pair_bias") is not None:  # the reference's cached projection
        assert rel_err(bias, squeeze(kw["pair_bias"], 3)) < 1e-5
    return a, s, bias, local


def _attention_pair_bias(m, rec):
    a, s, bias, local = _apb_inputs(m, rec["kwargs"])
    out = m.local(a, s, bias, no_padding(a)) if local else m(a, s, bias)
    return [(out, squeeze(rec["output"], 2))]


def _transition_block(m, rec):
    return [
        (m(squeeze(get(rec, 0, "a"), 2), squeeze(get(rec, 1, "s"), 2)), squeeze(rec["output"], 2))
    ]


def _diffusion_block(m, rec):
    kw = dict(rec["kwargs"], a=rec["args"][0], s=rec["args"][1], z=rec["args"][2])
    a, s, bias, local = _apb_inputs(m.attention_pair_bias, kw)
    out = m.local(a, s, bias, no_padding(a)) if local else m(a, s, bias)
    return [(out, squeeze(rec["output"][0], 2))]


def _diffusion_transformer(m, rec):
    kw = rec["kwargs"]
    local = bool(kw.get("n_queries"))
    biases = m.pair_biases(squeeze(kw["z"], 4 if local else 3))
    a, s = squeeze(kw["a"], 2), squeeze(kw["s"], 2)
    out = m.local(a, s, biases, no_padding(a)) if local else m(a, s, biases)
    return [(out, squeeze(rec["output"], 2))]


def _atom_transformer(m, rec):
    q, c, p = (squeeze(x, n) for x, n in zip(rec["args"], (2, 2, 4)))
    return [(m.local(q, c, m.pair_biases(p), no_padding(q)), squeeze(rec["output"], 2))]


def _atoms() -> AtomFeatures:
    return features().atoms


def _encoder(m, rec):
    kw = rec["kwargs"]
    atoms, num_tokens = _atoms(), features().num_tokens
    if kw.get("r_l") is None:
        cond = m.conditioning(atoms)
        q = cond.c
    else:
        cond = m.conditioning(atoms, squeeze(kw["s"], 2), squeeze(kw["z"], 3))
        q = cond.c + m.linear_no_bias_r(squeeze(kw["r_l"], 2))
    biases = m.atom_transformer.pair_biases(cond.p)
    a, q = m(q, cond, biases, atoms.atom_to_token, atoms.atom_mask, num_tokens)
    a_ref, q_ref, c_ref, p_ref = rec["output"]
    return [(a, squeeze(a_ref, 2)), (q, squeeze(q_ref, 2)),
            (cond.c, squeeze(c_ref, 2)), (cond.p, squeeze(p_ref, 4))]  # fmt: skip


def _decoder(m, rec):
    kw = rec["kwargs"]
    p = squeeze(kw["p_skip"], 4)
    q_skip = squeeze(kw["q_skip"], 2)
    out = m(squeeze(kw["a"], 2), q_skip, squeeze(kw["c_skip"], 2),
            m.atom_transformer.pair_biases(p), jnp.asarray(kw["atom_to_token_idx"]),
            no_padding(q_skip))  # fmt: skip
    return [(out, squeeze(rec["output"], 2))]


def _diffusion_conditioning(m, rec):
    kw = rec["kwargs"]
    sigma = squeeze(rec["args"][0], 0)
    single = m.single(m.single_base(kw["s_trunk"], kw["s_inputs"]), sigma)
    pair = m.pair(rec["args"][1], run()["trunk"]["z"])
    return [(single, squeeze(rec["output"][0], 2)), (pair, rec["output"][1])]


def _context(m):
    inputs, state = trunk_from_reference()
    return m.context(features().atoms, relative_position_features(features().tokens),
                     inputs.s_inputs, state.s, state.z)  # fmt: skip


def _diffusion_module(m, rec):
    kw = rec["kwargs"]
    x0 = m.denoise(_context(m), squeeze(kw["x_noisy"], 2), squeeze(kw["t_hat_noise_level"], 0))
    return [(x0, squeeze(rec["output"], 2))]


def _fourier(m, rec):
    t = squeeze(rec["kwargs"]["t_hat_noise_level"], 0)
    return [(m(t), squeeze(rec["output"], 1))]


def _relpe(m, rec):
    relp = rec["args"][0]
    return [(relative_position_features(features().tokens), relp), (m(relp), rec["output"])]


def _input_embedder(m, rec):
    return [(m(features().atoms, features().tokens), rec["output"])]


def _msa_block(m, rec):
    msa, z = get(rec, 0, "m"), get(rec, 1, "z")
    m_out, z_out = m(msa, z, jnp.ones(msa.shape[0], bool))
    pairs = [(z_out, rec["output"][1])]
    return pairs + ([(m_out, rec["output"][0])] if rec["output"][0] is not None else [])


def _msa_features(d) -> MSAFeatures:
    return MSAFeatures(
        msa=jax.nn.one_hot(jnp.asarray(d["msa"], jnp.int32), 32),
        has_deletion=jnp.asarray(d["has_deletion"]),
        deletion_value=jnp.asarray(d["deletion_value"]),
    )


def _msa_module(m, rec):
    d = rec["args"][0]
    msa = _msa_features(d)
    z = m(msa, rec["args"][1], rec["args"][2], jnp.ones(msa.msa.shape[0], bool))
    return [(z, rec["output"])]


def _template_features(d) -> TemplateFeatures:
    return TemplateFeatures(
        aatype=jnp.asarray(d["template_aatype"], jnp.int32),
        distogram=jnp.asarray(d["template_distogram"]),
        pseudo_beta_mask=jnp.asarray(d["template_pseudo_beta_mask"]),
        unit_vector=jnp.asarray(d["template_unit_vector"]),
        backbone_frame_mask=jnp.asarray(d["template_backbone_frame_mask"]),
    )


def _template_embedder(m, rec):
    d = rec["args"][0]
    u = m(_template_features(d), rec["args"][1], jnp.asarray(d["asym_id"], jnp.int32))
    return [(u, rec["output"])]


def _block(m, rec):
    s = rec["kwargs"]["s"] if "s" in rec["kwargs"] else rec["args"][0]
    z = rec["kwargs"]["z"] if "z" in rec["kwargs"] else rec["args"][1]
    s_out, z_out = m(s, z)
    return [(z_out, rec["output"][1])] + ([] if s is None else [(s_out, rec["output"][0])])


def _confidence_head(m, rec):
    kw = rec["kwargs"]
    f = features()
    coords = squeeze(kw["x_pred_coords"], 2)
    # The reference's torch.cdist uses the |a|²+|b|²-2ab expansion, off by ~1e-2 Å here;
    # feed both implementations the same distances (see test_confidence_fp32_floor).
    with mock.patch.object(jol1.confidence, "pairwise_distances", torch_cdist):
        out = m(kw["s_inputs"], kw["s_trunk"], kw["z_trunk"], coords, f.tokens.rep_atom,
                f.atoms.atom_to_token, f.atoms.atom_to_tokatom)  # fmt: skip
    names = ["plddt_logits", "pae_logits", "pde_logits", "resolved_logits"]
    return [(getattr(out, n), squeeze(r, r.ndim - 1)) for n, r in zip(names, rec["output"])]


def torch_cdist(x):
    x = torch.from_numpy(np.asarray(x))
    return jnp.asarray(torch.cdist(x, x).numpy())


ADAPTERS = {
    "Linear": _unary,
    "OpenfoldLinear": _unary,
    "BiasInitLinear": _unary,
    "OpenFoldLayerNorm": _unary,
    "FusedLayerNorm": _unary,
    "Transition": _unary,
    "Sequential": _unary,
    "AdaptiveLayerNorm": _adaln,
    "TriangleMultiplicationOutgoing": _unary,
    "TriangleMultiplicationIncoming": _unary,
    "TriangleAttention": _tri_attention,
    "mol.model.triangular.layers.Attention": _openfold_attention,
    "mol.model.modules.primitives.Attention": _attention,
    "AttentionPairBias": _attention_pair_bias,
    "ConditionedTransitionBlock": _transition_block,
    "DiffusionTransformerBlock": _diffusion_block,
    "DiffusionTransformer": _diffusion_transformer,
    "AtomTransformer": _atom_transformer,
    "AtomAttentionEncoder": _encoder,
    "AtomAttentionDecoder": _decoder,
    "DiffusionConditioning": _diffusion_conditioning,
    "DiffusionModule": _diffusion_module,
    "FourierEmbedding": _fourier,
    "RelativePositionEncoding": _relpe,
    "InputFeatureEmbedder": _input_embedder,
    "OuterProductMean": lambda m, rec: [
        (m(rec["args"][0], jnp.ones(rec["args"][0].shape[0], bool)), rec["output"])
    ],
    "MSAPairWeightedAveraging": lambda m, rec: [(m(*rec["args"]), rec["output"])],
    "MSAStack": lambda m, rec: [(m(*rec["args"]), rec["output"])],
    "MSABlock": _msa_block,
    "MSAModule": _msa_module,
    "TemplateEmbedder": _template_embedder,
    "PairformerBlock": _block,
    "PairMixerBlock": _block,
    "PairformerStack": _block,
    "DistogramHead": _unary,
    "ConfidenceHead": _confidence_head,
}
# Parameter-free modules with no JAX module of their own, and the width adapter (a Linear
# inside PairformerStack).
SKIPPED = {"Sigmoid", "Identity", "ReLU", "Softmax", "PairformerWidthAdapter"}
# Measured fp32 floors: the confidence stack's single track is ~1e-5 from a float64 run
# for torch and JAX alike (test_confidence_fp32_floor).
TOLERANCE = {"confidence_head": 5e-5, "confidence_head.pairformer_stack": 5e-5}


def captured_modules() -> list[str]:
    return sorted(p.stem for p in (CAPTURES / "modules").glob("*.pkl"))


@pytest.mark.parametrize("name", captured_modules())
def test_module(name):
    torch_module = reference().get_submodule(name)
    kind = type(torch_module)
    adapter = ADAPTERS.get(f"{kind.__module__}.{kind.__name__}") or ADAPTERS.get(kind.__name__)
    if kind.__name__ in SKIPPED:
        pytest.skip(f"{kind.__name__} has no JAX module")
    assert adapter is not None, f"no adapter for {kind.__module__}.{kind.__name__}"
    with torch.no_grad():
        jax_module = from_torch(torch_module)
    for ours, ref in adapter(jax_module, capture(name)):
        check(ours, ref, TOLERANCE.get(name, 1e-5))


# ---------------------------------------------------------------- end to end


@cache
def trunk_from_reference() -> tuple[InputEmbedding, TrunkState]:
    """Torch trunk outputs, to test later stages in isolation."""
    trunk = run()["trunk"]
    inputs = eqx.filter_jit(lambda m, f: m.embed(f))(model(), features())
    inputs = InputEmbedding(jnp.asarray(trunk["s_inputs"]), inputs.s_init, inputs.z_init)
    return inputs, TrunkState(s=jnp.asarray(trunk["s"]), z=jnp.asarray(trunk["z"]))


def test_trunk():
    trunk = run()["trunk"]
    inputs = eqx.filter_jit(lambda m, f: m.embed(f))(model(), features())
    check(inputs.s_inputs, trunk["s_inputs"], 1e-6)

    @eqx.filter_jit
    def run_trunk(model, features, inputs):
        # The capture uses the full MSA every cycle (tests/capture.py).
        state = TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))
        all_rows = jnp.ones(features.msa.msa.shape[0], bool)
        for _ in range(NUM_CYCLES):
            state = model.cycle(features, inputs, state, all_rows)
        return state

    state = run_trunk(model(), features(), inputs)
    check(state.s, trunk["s"], 1e-5)
    check(state.z, trunk["z"], 1e-5)
    check(model().distogram_logits(state), capture("distogram_head")["output"], 1e-5)


def test_denoise():
    """The first denoiser call of the sampling run (highest noise level)."""
    inputs, state = trunk_from_reference()
    context = eqx.filter_jit(lambda m, f, i, s: m.diffusion_context(f, i, s))(
        model(), features(), inputs, state
    )
    rec = capture("diffusion_module")["kwargs"]
    x0 = eqx.filter_jit(lambda m, c, x, t: m.denoise(c, x, t))(
        model(), context, squeeze(rec["x_noisy"], 2), squeeze(rec["t_hat_noise_level"], 0)
    )
    check(x0, squeeze(capture("diffusion_module")["output"], 2), 1e-5)


def test_confidence_fp32_floor():
    """The confidence stack's single track is ill-conditioned in fp32: torch and JAX are
    both ~1e-5 from a float64 evaluation, and JAX is no worse than torch."""
    rec = capture("confidence_head.pairformer_stack")
    s, z = rec["args"][:2]
    stack = model().confidence_head.pairformer_stack
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        f64 = jax.tree.map(lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, stack)
        exact = eqx.filter_jit(lambda m, s, z: m(s, z))(f64, jnp.asarray(s, jnp.float64),
                                                        jnp.asarray(z, jnp.float64))  # fmt: skip
    ours = eqx.filter_jit(lambda m, s, z: m(s, z))(stack, s, z)
    for i in range(2):
        torch_err, jax_err = rel_err(rec["output"][i], exact[i]), rel_err(ours[i], exact[i])
        print(f"vs float64: torch {torch_err:.1e}, jax {jax_err:.1e}")
        assert jax_err < 2 * torch_err + 1e-6


def test_template_embedder_with_templates():
    """The capture only has empty (all-gap) templates; drive both with real-looking ones."""
    rng = np.random.default_rng(0)
    T, Nt = features().num_tokens, 2
    d = {
        "asym_id": run()["features"]["asym_id"],
        "template_aatype": rng.integers(0, 32, (Nt, T)),
        "template_distogram": np.eye(39, dtype=np.float32)[rng.integers(0, 39, (Nt, T, T))],
        "template_pseudo_beta_mask": (rng.random((Nt, T, T)) < 0.8).astype(np.float32),
        "template_unit_vector": rng.normal(size=(Nt, T, T, 3)).astype(np.float32),
        "template_backbone_frame_mask": (rng.random((Nt, T, T)) < 0.8).astype(np.float32),
    }
    z = rng.normal(size=(T, T, 448)).astype(np.float32)
    with torch.no_grad():
        ref = (
            reference()
            .template_embedder(
                {k: torch.from_numpy(np.asarray(v)) for k, v in d.items()}, torch.from_numpy(z)
            )
            .numpy()
        )
    ours = model().template_embedder(
        _template_features(d), jnp.asarray(z), jnp.asarray(d["asym_id"], jnp.int32)
    )
    check(ours, ref, 1e-5)


def test_msa_row_mask_is_row_removal():
    """Masking MSA rows out is exactly the MSA module on the remaining rows."""
    f, (inputs, state) = features(), trunk_from_reference()
    keep = np.array(jax.random.bernoulli(jax.random.key(0), 0.5, (f.msa.msa.shape[0],)))
    keep[0] = True
    msa_module = eqx.filter_jit(lambda m, msa, z, s, mask: m(msa, z, s, mask))
    masked = msa_module(model().msa_module, f.msa, state.z, inputs.s_inputs, jnp.asarray(keep))
    subset = jax.tree.map(lambda x: x[keep], f.msa)
    all_rows = jnp.ones(int(keep.sum()), bool)
    removed = msa_module(model().msa_module, subset, state.z, inputs.s_inputs, all_rows)
    check(masked, removed, 1e-5)


def test_gradients():
    """Gradients reach the sequence through the trunk, the denoiser and the confidence
    head, including the confidence head's direct trunk inputs."""
    f = features()

    @eqx.filter_jit
    def loss(restype, model, f):
        f = eqx.tree_at(lambda f: f.tokens.restype, f, restype)
        inputs = model.embed(f)
        state = model.trunk(f, inputs, num_cycles=2, key=jax.random.key(0))
        context = model.diffusion_context(f, inputs, state)
        x = model.denoise(context, 10.0 * f.atoms.ref_pos, jnp.float32(10.0))
        conf = model.confidence(f, inputs, state, x)
        return (x**2).mean() + jax.nn.logsumexp(conf.plddt_logits, -1).mean()

    g = jax.grad(loss)(f.tokens.restype, model(), f)
    assert np.isfinite(g).all() and np.abs(g).max() > 0


def test_atom_padding_is_inert():
    """Padding atoms appended to the atom axis do not influence the real atoms: input
    embedding, diffusion context and denoising agree exactly whatever the padding holds (on
    CPU: GPU scatter-adds are nondeterministic at the ulp level), and match the unpadded
    system to fp32 rounding."""
    f, (_, state) = features(), trunk_from_reference()
    N, pad = f.num_atoms, 45
    atoms = f.atoms

    def padded(x, fill):
        return jnp.concatenate([x, jnp.asarray(fill, x.dtype)])

    def with_padding(seed):
        rng = np.random.default_rng(seed)
        junk = AtomFeatures(
            ref_pos=padded(atoms.ref_pos, rng.normal(size=(pad, 3)) * 10),
            ref_charge=padded(atoms.ref_charge, np.zeros(pad)),
            ref_mask=padded(atoms.ref_mask, np.zeros(pad)),
            ref_element=padded(atoms.ref_element, np.eye(128)[rng.integers(0, 128, pad)]),
            ref_atom_name_chars=padded(atoms.ref_atom_name_chars, np.zeros((pad, 256))),
            ref_space_uid=padded(atoms.ref_space_uid, np.zeros(pad)),
            atom_to_token=padded(atoms.atom_to_token, rng.integers(0, f.num_tokens, pad)),
            atom_to_tokatom=padded(atoms.atom_to_tokatom, np.zeros(pad)),
            atom_mask=padded(atoms.atom_mask, np.zeros(pad, bool)),
        )
        return eqx.tree_at(lambda f: f.atoms, f, junk), rng.normal(size=(pad, 3)) * 30

    @eqx.filter_jit
    def run(model, f, state, x):
        inputs = model.embed(f)
        context = model.diffusion_context(f, inputs, state)
        return inputs.s_inputs, model.denoise(context, x, jnp.float32(30.0))

    x = jnp.asarray(np.random.default_rng(0).normal(size=(N, 3)) * 30, jnp.float32)
    (g1, x1), (g2, x2) = with_padding(1), with_padding(2)
    with jax.default_device(jax.devices("cpu")[0]):
        cpu = jax.devices("cpu")[0]
        m, state = jax.device_put((model(), state), cpu)
        s_ref, x0_ref = run(m, jax.device_put(f, cpu), state, x)
        s1, x0_1 = run(m, jax.device_put(g1, cpu), state, padded(x, x1))
        s2, x0_2 = run(m, jax.device_put(g2, cpu), state, padded(x, x2))
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(x0_1[:N], x0_2[:N])
    check(s1, s_ref, 1e-5)
    check(x0_1[:N], x0_ref, 1e-5)
