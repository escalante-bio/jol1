"""Record torch ground truth for the parity tests (``reference`` extra, CPU, true fp32).

Runs the reference ``Mol`` model once on ``tests/data/system.json`` and writes, under
``tests/captures/``:

- ``modules/<name>.pkl``: input and output of the first call of every module (repeated
  stack blocks only for a few indices),
- ``run.pkl``: the feature dict, the trunk outputs, every random draw of the sampler,
  the sampled coordinates and the confidence logits.

The MSA is used in full (the reference subsamples rows at random every cycle).

    uv run --extra reference python tests/capture.py
"""

from __future__ import annotations

import copy
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

os.environ.setdefault("LAYERNORM_TYPE", "torch")  # before the reference is imported
ROOT = Path(__file__).resolve().parent
NUM_CYCLES, NUM_STEPS = 3, 20
KEEP_BLOCKS = {0, 1, 8, 23, 47}  # stack block indices whose calls are recorded


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().copy()  # .numpy() shares memory with x
    if isinstance(x, (list, tuple)):
        return type(x)(to_np(v) for v in x)
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if x is None or isinstance(x, (int, float, bool, str)):
        return x
    return f"<{type(x).__name__}>"


def main(out: Path) -> None:
    sys.path.insert(0, str(ROOT.parent / "src"))
    import json

    import mol.model.generator as generator
    import mol.model.utils as mol_utils

    from jol1.convert import load_reference_model
    from jol1.reference import reference_feature_dict

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    np.random.seed(0)

    sample = json.loads((ROOT / "data" / "system.json").read_text())
    feats, _ = reference_feature_dict(sample, use_msa=True, data_dir=ROOT.parent / "mol_data")
    model = load_reference_model(ROOT.parent / "mol_data/checkpoint/mol-1-base.pt")
    model.N_cycle = NUM_CYCLES
    model.configs.sample_diffusion["N_step"] = NUM_STEPS
    model.configs.sample_diffusion["N_sample"] = 1

    shutil.rmtree(out, ignore_errors=True)
    (out / "modules").mkdir(parents=True)
    seen: set[str] = set()
    inputs: dict[str, tuple] = {}

    # Inputs are copied before the call: some reference modules update them in place.
    def pre_hook(name):
        def fn(module, args, kwargs):
            if name not in seen and name not in inputs:
                inputs[name] = (to_np(args), to_np(kwargs))

        return fn

    def hook(name):
        def fn(module, args, kwargs, output):
            if name in seen:
                return
            seen.add(name)
            args, kwargs = inputs.pop(name)
            record = {"type": type(module).__name__, "args": args, "kwargs": kwargs,
                      "output": to_np(output)}  # fmt: skip
            with open(out / "modules" / f"{name}.pkl", "wb") as f:
                pickle.dump(record, f)

        return fn

    for name, module in model.named_modules():
        indices = {int(i) for i in re.findall(r"\.(\d+)(?:\.|$)", name)}
        if name and indices <= KEEP_BLOCKS:
            module.register_forward_pre_hook(pre_hook(name), with_kwargs=True)
            module.register_forward_hook(hook(name), with_kwargs=True)

    run: dict = {"features": feats, "randn": [], "rotations": []}

    def full_msa(n, device=torch.device("cpu"), lower_bound=1, strategy="random"):
        return torch.arange(n, device=device)

    trunk_fn = model.get_pairformer_output

    def record_trunk(*args, **kwargs):
        s_inputs, s, z = trunk_fn(*args, **kwargs)
        run["trunk"] = to_np({"s_inputs": s_inputs, "s": s, "z": z})
        return s_inputs, s, z

    randn, rotation = torch.randn, mol_utils.uniform_random_rotation

    def record_randn(*args, **kwargs):
        x = randn(*args, **kwargs)
        run["randn"].append(x.numpy().copy())
        return x

    def record_rotation(*args, **kwargs):
        r = rotation(*args, **kwargs)
        run["rotations"].append(r.numpy().copy())
        return r

    with (
        torch.no_grad(),
        mock.patch.object(mol_utils, "sample_indices", full_msa),
        mock.patch.object(model, "get_pairformer_output", record_trunk),
        mock.patch.object(generator.torch, "randn", record_randn),
        mock.patch.object(mol_utils, "uniform_random_rotation", record_rotation),
    ):
        input_dict = {k: torch.from_numpy(np.array(v)) for k, v in copy.deepcopy(feats).items()}
        pred, _, _ = model(
            input_feature_dict=input_dict,
            label_full_dict=None,
            label_dict=None,
            mode="inference",
            disable_inplace=True,
            mc_dropout_apply_rate=0.0,
        )
    run["prediction"] = to_np({k: pred[k] for k in ("coordinate", "plddt", "pae", "pde",
                                                     "resolved", "contact_probs")})  # fmt: skip
    run["schedule"] = to_np(model.inference_noise_scheduler(N_step=NUM_STEPS))
    with open(out / "run.pkl", "wb") as f:
        pickle.dump(run, f)
    print(f"wrote {len(seen)} module captures and run.pkl to {out}")


if __name__ == "__main__":
    main(ROOT / "captures")
