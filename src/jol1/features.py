"""Typed, unbatched model inputs.

``Features`` holds exactly the per-system arrays the network reads. It is built from the
reference featurizer's output dict by :meth:`Features.from_reference_dict`, the single
boundary between dict-keyed reference features and the typed JAX model (and the only
place reference key names appear).
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jol1.embedders import TokenFeatures
from jol1.pairformer import MSAFeatures, TemplateFeatures
from jol1.transformer import AtomFeatures


class Features(eqx.Module):
    """One system's inputs: ``T`` tokens, ``N`` atoms, ``S`` MSA rows, ``Nt`` templates."""

    tokens: TokenFeatures
    atoms: AtomFeatures
    msa: MSAFeatures
    templates: TemplateFeatures

    @property
    def num_tokens(self) -> int:
        return self.tokens.restype.shape[0]

    @property
    def num_atoms(self) -> int:
        return self.atoms.ref_pos.shape[0]

    @classmethod
    def from_reference_dict(cls, feats: Mapping[str, np.ndarray]) -> Features:
        """Build from the reference ``input_feature_dict`` (numpy, unbatched)."""

        def get(key, dtype):
            return jnp.asarray(np.asarray(feats[key]), dtype=dtype)

        i32, f32 = jnp.int32, jnp.float32
        atom_to_token = np.asarray(feats["atom_to_token_idx"])
        num_tokens = np.asarray(feats["restype"]).shape[0]
        rep_atom = np.flatnonzero(np.asarray(feats["distogram_rep_atom_mask"]))
        if not np.array_equal(atom_to_token[rep_atom], np.arange(num_tokens)):
            raise ValueError("expected exactly one distogram representative atom per token")
        return cls(
            tokens=TokenFeatures(
                restype=get("restype", f32),
                profile=get("profile", f32),
                deletion_mean=get("deletion_mean", f32),
                asym_id=get("asym_id", i32),
                residue_index=get("residue_index", i32),
                entity_id=get("entity_id", i32),
                sym_id=get("sym_id", i32),
                token_index=get("token_index", i32),
                token_bonds=get("token_bonds", f32),
                rep_atom=jnp.asarray(rep_atom, i32),
            ),
            atoms=AtomFeatures(
                ref_pos=get("ref_pos", f32),
                ref_charge=get("ref_charge", f32),
                ref_mask=get("ref_mask", f32),
                ref_element=get("ref_element", f32),
                ref_atom_name_chars=get("ref_atom_name_chars", f32).reshape(-1, 4 * 64),
                ref_space_uid=get("ref_space_uid", i32),
                atom_to_token=jnp.asarray(atom_to_token, i32),
                atom_to_tokatom=get("atom_to_tokatom_idx", i32),
                atom_mask=jnp.ones(atom_to_token.shape, bool),  # the reference never pads
            ),
            msa=MSAFeatures(
                msa=jax.nn.one_hot(get("msa", i32), 32),
                has_deletion=get("has_deletion", f32),
                deletion_value=get("deletion_value", f32),
            ),
            templates=TemplateFeatures(
                aatype=get("template_aatype", i32),
                distogram=get("template_distogram", f32),
                pseudo_beta_mask=get("template_pseudo_beta_mask", f32),
                unit_vector=get("template_unit_vector", f32),
                backbone_frame_mask=get("template_backbone_frame_mask", f32),
            ),
        )
