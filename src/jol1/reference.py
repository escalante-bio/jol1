"""Featurization via the upstream Mol-1 package (``reference`` extra).

Building model inputs (tokenization, CCD reference conformers, MSA and template
features) stays on the reference implementation's numpy/RDKit/biotite tooling; only the
network runs in JAX. Featurization needs the reference's CCD data (``common/`` of the
Mol-1 data release, see ``scripts/database/download_mol_data.sh``), passed as
``data_dir``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from jol1.features import Features

os.environ.setdefault("LAYERNORM_TYPE", "torch")


def reference_feature_dict(
    sample: Mapping[str, Any], *, use_msa: bool, data_dir: Path
) -> tuple[dict[str, np.ndarray], Any]:
    """The reference ``input_feature_dict`` (as numpy) and biotite ``AtomArray`` for one
    entry of a ``mol pred`` input JSON. ``data_dir`` is the Mol-1 data root (containing
    ``common/components.cif``). MSA paths in the JSON are read as given; without
    ``use_msa`` a single-sequence MSA is used. Templates are left empty (as ``mol pred``
    does by default)."""
    from mol.data.core import ccd
    from mol.data.inference.infer_dataloader import InferenceDataset

    # The reference resolves these from $MOL_ROOT_DIR at import time; featurization reads
    # only these two module globals, so point them at ``data_dir`` instead.
    common = Path(data_dir) / "common"
    if not (common / "components.cif").is_file():
        raise FileNotFoundError(f"no CCD components file under {common}")

    dataset = InferenceDataset.__new__(InferenceDataset)  # skip config-driven setup
    dataset.use_msa = use_msa
    dataset.msa_pair_as_unpair = True
    dataset.use_rna_msa = False
    dataset.use_template = False
    dataset.online_template_featurizer = None
    with (
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        data, atom_array, _ = dataset.process_one(dict(sample))
    feats = {k: v.numpy() for k, v in data["input_feature_dict"].items() if hasattr(v, "numpy")}
    return feats, atom_array


def featurize(sample: Mapping[str, Any], *, use_msa: bool, data_dir: Path) -> tuple[Features, Any]:
    """Model inputs and the biotite ``AtomArray`` (atom order of the coordinates) for one
    entry of a ``mol pred`` input JSON; see :func:`reference_feature_dict`."""
    feats, atom_array = reference_feature_dict(sample, use_msa=use_msa, data_dir=data_dir)
    return Features.from_reference_dict(feats), atom_array


def write_cif(atom_array: Any, coords: np.ndarray, path: str) -> None:
    """Write sampled coordinates (``[N, 3]``, atom order of :func:`featurize`) as mmCIF."""
    from biotite.structure.io import pdbx

    atom_array = atom_array.copy()
    atom_array.coord = np.asarray(coords, dtype=np.float32)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atom_array)
    cif.write(path)
