"""Pure-JAX unit tests (no torch, no checkpoint)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jol1 import sample_msa_rows
from jol1.confidence import pairwise_distances
from jol1.embedders import TokenFeatures, relative_position_features
from jol1.transformer import H, W, key_window_index, mean_atoms_to_tokens, window_keys


def reference_key_windows(x: np.ndarray) -> np.ndarray:
    """The reference's pad + unfold (``rearrange_qk_to_dense_trunk``), in numpy."""
    n = len(x)
    n_trunks = -(-n // W)
    pad_left = (H - W) // 2
    pad_right = int((n_trunks - 1 / 2) * W + H / 2 - n + 1 / 2)
    padded = np.pad(x, (pad_left, pad_right))
    return np.stack([padded[k * W : k * W + H] for k in range(n_trunks)])


@pytest.mark.parametrize("num_atoms", [1, 31, 32, 33, 128, 300, 1000])
def test_key_windows_match_reference(num_atoms):
    x = np.arange(1, num_atoms + 1, dtype=np.float32)
    np.testing.assert_array_equal(window_keys(jnp.asarray(x)), reference_key_windows(x))
    index, in_range = key_window_index(num_atoms)
    np.testing.assert_array_equal(in_range, reference_key_windows(x) > 0)
    # Every query atom sees itself.
    for atom in range(num_atoms):
        assert atom in index[atom // W][in_range[atom // W]]


def test_mean_atoms_to_tokens():
    x = jnp.arange(10.0)[:, None]
    tok = jnp.array([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
    real = jnp.ones(10, bool)
    np.testing.assert_allclose(mean_atoms_to_tokens(x, tok, real, 4)[:, 0], [0.5, 3.0, 0.0, 7.0])
    real = real.at[9].set(False)  # padding atoms do not count
    np.testing.assert_allclose(mean_atoms_to_tokens(x, tok, real, 4)[:, 0], [0.5, 3.0, 0.0, 6.5])


def test_sample_msa_rows():
    masks = jax.vmap(lambda k: sample_msa_rows(k, 10, 16384))(
        jax.random.split(jax.random.key(0), 2000)
    )
    sizes = np.asarray(masks.sum(-1))
    assert sizes.min() == 1 and sizes.max() == 10  # uniform over [1, 10]
    assert abs(np.bincount(sizes, minlength=11)[1:].std() / 200) < 0.2
    capped = sample_msa_rows(jax.random.key(1), 50, 4)
    assert int(capped.sum()) <= 4


def test_relative_position_features():
    T = 6
    tokens = TokenFeatures(
        restype=jnp.zeros((T, 32)),
        profile=jnp.zeros((T, 32)),
        deletion_mean=jnp.zeros(T),
        asym_id=jnp.array([0, 0, 0, 1, 1, 2]),
        residue_index=jnp.array([1, 2, 2, 1, 2, 1]),
        entity_id=jnp.array([0, 0, 0, 0, 0, 1]),
        sym_id=jnp.array([0, 0, 0, 1, 1, 0]),
        token_index=jnp.arange(T),
        token_bonds=jnp.zeros((T, T)),
        rep_atom=jnp.arange(T),
    )
    relp = relative_position_features(tokens)
    assert relp.shape == (T, T, 139)
    np.testing.assert_array_equal(relp.sum(-1), 3 + relp[..., 132])  # three one-hots + bit
    assert relp[0, 1, 32 - 1] == 1  # residue offset -1 within a chain
    assert relp[0, 3, 65] == 1  # different chains: "far" bin
    assert relp[0, 3, 132] == 1 and relp[0, 5, 132] == 0  # same entity
    assert relp[0, 3, 133 + 2 - 1] == 1  # chain offset -1 within an entity


def test_pairwise_distances_gradient():
    x = jax.random.normal(jax.random.key(0), (5, 3))
    d = pairwise_distances(x)
    np.testing.assert_allclose(d, np.linalg.norm(x[:, None] - x[None], axis=-1), rtol=1e-6)
    g = jax.grad(lambda x: pairwise_distances(x).sum())(x)
    assert np.isfinite(g).all()
