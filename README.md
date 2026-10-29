# jol1

A JAX/[Equinox](https://github.com/patrick-kidger/equinox) port of
[Mol-1](https://github.com/molence/Mol-1) (reference commit `7480cc4`, checkpoint
`mol-1-base`).

Mol-1 is a Protenix-derived all-atom structure predictor (577M parameters): an AF3 input
embedder, a recycled trunk (template embedder, 4-block MSA module, then 8 PairMixer blocks
at pair width 448 followed by 40 Pairformer blocks at width 320), the AF3 diffusion module
and a 4-block confidence head.

## What is ported

| Ported (JAX) | Kept on the reference (`reference` extra) | Not ported |
|---|---|---|
| Input embedder, relative position encoding | Featurization (JSON → `Features`), MSA / template search | Training, losses, permutation |
| Template embedder, MSA module, PairMixer/Pairformer stack, recycling | CIF output | Summary confidence metrics (pTM, ipTM, ranking) |
| Diffusion module (step-invariant context + EDM denoiser) | | MC dropout (off in `mol pred`) |
| Distogram and confidence heads | | Custom kernels (cuEquivariance, FlashPairformer, fused LayerNorm) — same math in XLA |
| Reference MSA row subsampling (as a row mask, always on in `trunk`) | | |
| Weight conversion, torch-free save/load | | Diffusion sampler loop (see `Mol1.sampler_settings`) |

`import jol1` never imports torch; only `jol1.convert` and `jol1.reference` do.

## Install

```bash
uv sync --extra reference --extra cuda   # CUDA 12 JAX + reference tooling (CPU torch, molence)
```

Weights and reference data come from the Mol-1 release. The usage below and the tests expect
them under `mol_data/` in the repository root:

```
mol_data/
  checkpoint/mol-1-base.pt   # https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt
  common/                    # CCD components: MOL_ROOT_DIR=mol_data bash scripts/database/download_mol_data.sh
```

(`download_mol_data.sh` is in the Mol-1 repository.) Featurization takes the data
directory as an argument (`data_dir`); only the tests assume this layout.

## Usage

```python
import jax
import jax.numpy as jnp

from jol1 import load_model, save_model
from jol1.convert import convert_checkpoint      # needs `reference`
from jol1.reference import featurize, write_cif   # needs `reference`

model = convert_checkpoint("mol_data/checkpoint/mol-1-base.pt")  # ~15 s
save_model(model, "mol_data/mol1_base")          # later: load_model(...), torch-free

entry = {"name": "x", "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQR", "count": 1}}]}
features, atom_array = featurize(entry, use_msa=False, data_dir="mol_data")

inputs = model.embed(features)                                   # once per system
state = model.trunk(features, inputs, num_cycles=20, key=jax.random.key(0))
context = model.diffusion_context(features, inputs, state)       # once per trunk output
x0 = model.denoise(context, x_noisy, jnp.float32(sigma))         # one EDM denoising step
confidence = model.confidence(features, inputs, state, coords)
write_cif(atom_array, coords, "x.cif")
```

`denoise` is an EDM x0-predictor; sample with any AF3-style (Algorithm 18) sampler, using
`model.sampler_settings` for the reference's schedule and churn parameters (see below).
Wrap stages in `eqx.filter_jit` and pass the model as an argument (never close over it).
Everything is unbatched (`T` tokens, `N` atoms, `S` MSA rows); `jax.vmap` over keys for
several samples.

### API

```python
Mol1.embed(features) -> InputEmbedding                         # s_inputs, s_init, z_init
Mol1.trunk(features, inputs, *, num_cycles, key=None, msa_rows=None) -> TrunkState  # s, z
Mol1.cycle(features, inputs, state, msa_row_mask) -> TrunkState      # one recycle
Mol1.distogram_logits(state) -> f32[T, T, 64]
Mol1.diffusion_context(features, inputs, state) -> DiffusionContext
Mol1.denoise(context, x: f32[N, 3], sigma: f32[]) -> f32[N, 3]       # EDM x0, Å
Mol1.confidence(features, inputs, state, coords) -> ConfidenceOutput  # pLDDT/PAE/PDE/resolved logits
Mol1.sampler_settings                                           # checkpoint's schedule + Alg. 18 params
sample_msa_rows(key, num_rows, max_rows) -> bool[S]

Features.from_reference_dict(feats)          # boundary adapter from the reference feature dict
Features.msa.msa: f32[S, T, 32]               # one-hot (or soft) MSA
Features.atoms.atom_mask: bool[N]             # False for padding atoms (never attended to)
jol1.reference.featurize(entry, *, use_msa, data_dir) -> (Features, AtomArray)
jol1.convert.convert_checkpoint(path) -> Mol1
save_model(model, path) / load_model(path)
```

### Notes on the reference's behaviour

- **Recycling.** `num_cycles` counts trunk passes (`mol pred` uses 20). Only the last pass
  is differentiated; earlier outputs are `stop_gradient`-ed.
- **MSA subsampling.** At inference the reference draws a fresh random MSA subset every
  cycle (size uniform in `[1, S]`). Rows only interact through the outer-product mean, so
  a row mask reproduces this exactly with static shapes. `trunk` subsamples when given a
  `key`; without one it uses every row each cycle (deterministic, e.g. for design).
  `msa_rows` keeps only the leading rows (the query first), which bounds the cost of the
  MSA module; call `cycle` with your own row masks to choose the rows.
- **Templates** always run: without template hits the featurizer emits four all-gap
  templates, and the embedder averages over them, as in the reference.
- **Diffusion caching.** Pair conditioning, atom conditioning and every attention pair
  bias are step-invariant and computed once in `diffusion_context` (the reference's
  `full` cache mode).
- **Sampler.** The reference's `sample_diffusion` uses a Karras schedule (σ_data 16,
  s_max 160, s_min 4e-4, ρ 7, last σ = 0), mean-centring plus a random rotation and
  translation every step, churn γ₀ = 0.8 while σ_next > 1, λ = 1.003, η = 1.5, and no
  alignment; these are in `Mol1.sampler_settings`. Atoms may be padded
  (`Features.atoms.atom_mask`): padding atoms never influence real atoms
  (`test_atom_padding_is_inert`), so centre on the real atoms only.
- **Confidence head.** The reference detaches its trunk inputs; the port does not, so
  confidence gradients reach the sequence directly as well as through the coordinates
  (forward values are unchanged). Distances are computed directly
  rather than with `torch.cdist`'s matmul expansion (which is off by ~1e-2 Å).

## Parity with the PyTorch reference

All in float32 with `"highest"` matmul precision, real converted `mol-1-base` weights,
torch on CPU, JAX on GPU. Test system (`tests/data/system.json`): a 48-residue homodimer
with a 94-row MSA, a 6-nt DNA strand, ATP and Mg²⁺ — 134 tokens, 936 atoms. Errors are
max |Δ| / max |reference|.

| Level | Max relative error |
|---|---|
| Every captured module call (965 calls of 36 module types, each torch submodule converted on its own) | ≤ 1e-5, confidence head ≤ 5e-5 (see below) |
| Input embedding `s_inputs` | 2.4e-8 |
| Trunk `s` / `z`, 3 cycles | 6.1e-7 / 1.7e-6 |
| Distogram logits | 1.1e-6 |
| Denoising step (σ = 4608 Å) | 2.4e-6 |
| Template embedder with non-empty (synthetic) templates | 7.6e-7 |
| Confidence logits: PAE / PDE / pLDDT / resolved | 3.6e-7 / 6.1e-7 / 1.9e-5 / 1.4e-5 |

Numerical notes:

- **Confidence single track.** The confidence Pairformer's single representation is
  ill-conditioned in fp32: against a float64 evaluation, torch is off by 9.9e-6 and JAX by
  8.4e-6 (`test_confidence_fp32_floor`), which is where the pLDDT/resolved errors come from.
- **`torch.cdist`.** The reference's distance matrix uses the |a|²+|b|²−2ab expansion and is
  off by up to 1.1e-2 Å on the test structure; the JAX port computes distances directly
  (5e-6 Å from float64). The confidence parity test feeds both the same distances; with
  their own distances PAE agrees to 3e-5.
- **MSA row masks** give the same result as removing the rows (1.0e-6).
- Gradients through the trunk, the denoiser and the confidence head are finite.

## Performance

H100 80 GB, JAX default (TF32) matmul precision, 448 tokens / 4195 atoms (7r6r protein as
a homodimer plus two DNA strands, single-sequence MSA):

| Stage | First call (incl. compile) | Steady state | Peak memory |
|---|---|---|---|
| `embed` + `trunk`, 20 cycles | 58 s | 33 s | 14.4 GiB |
| `diffusion_context` | 2.8 s | 0.02 s | |
| `confidence` | 3.0 s | 0.18 s | 16.1 GiB |
| `value_and_grad` through 4 cycles + denoise + confidence | 75 s | 12.7 s | 53.9 GiB |

Triangle attention dominates the trunk (~21 of 33 ms per Pairformer block): fp32 attention
materialises `[T, heads, T, T]` logits. The reference runs its trunk in bf16 by default; a
bf16 path with cuDNN flash attention is the obvious next optimisation.

## Tests

```bash
uv run pytest                                   # pure-JAX unit tests
uv run python tests/capture.py                  # record torch ground truth (~3 min, CPU)
uv run pytest -m reference                      # parity (needs captures + checkpoint)
```

## License

Apache-2.0, as upstream Mol-1 (Copyright 2026 Molence Bio), which derives from Protenix
(Copyright 2024 ByteDance) and OpenFold; see `NOTICE`. Model weights are distributed by
Molence Bio and are not redistributed here.
