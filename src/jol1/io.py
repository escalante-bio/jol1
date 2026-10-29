"""Torch-free save/load of converted models (array leaves + a pickled shape skeleton)."""

from __future__ import annotations

import pickle
from pathlib import Path

import equinox as eqx
import jax

from jol1.model import Mol1


def save_model(model: Mol1, path: Path) -> None:
    """Write ``<path>.eqx`` (weights) and ``<path>.skeleton.pkl`` (structure)."""
    path = Path(path)
    eqx.tree_serialise_leaves(path.with_suffix(".eqx"), model)
    skeleton = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x, model
    )
    with open(path.with_suffix(".skeleton.pkl"), "wb") as f:
        pickle.dump(skeleton, f)


def load_model(path: Path) -> Mol1:
    path = Path(path)
    with open(path.with_suffix(".skeleton.pkl"), "rb") as f:
        skeleton = pickle.load(f)
    return eqx.tree_deserialise_leaves(path.with_suffix(".eqx"), skeleton)
