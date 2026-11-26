# speedChem

Fast geometry optimization for semiempirical quantum chemistry. It started as the GPU QM engine
of [rhoMech](https://github.com/mfiaschi5/rhoMech), which depends on it.

- `speedchem.anc`: optimization in approximate normal coordinates with RFO steps (the scheme
  of xtb's ANCopt), on any energy-and-gradient function, in numpy.
- `speedchem.bonds`: the bond checks of a run on RDKit molecules: walls on the input bonds, a
  dissociation guard that stops a free run once a bond has stayed broken, and the final
  connectivity check.
- `speedchem.gpu`: GFN2-xTB with ALPB water on CUDA, exact to tblite 0.7.0 at ~1e-11 Hartree.
  It has the integrals, the batched SCF, energies and gradients, and `optimize.relax_many`:
  tblite-anc's optimization for many molecules at once, in batches that refill as molecules
  finish.

## Install

```bash
pip install "speedchem @ git+https://github.com/mfiaschi5/speedChem"
```

`speedchem.gpu` also needs torch with CUDA (bring your own build: `pip install
"speedchem[gpu]"` pulls the default one) and the nvcc of a CUDA 13 toolchain. The CUDA library
is compiled on first use. Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `KSXTB_NVCC` | `~/.local/share/ksxtb/tc13x/.../nvidia/cu13/bin/nvcc` | the nvcc to use |
| `KSXTB_ARCH` | `sm_89` | the GPU architecture to compile for |
| `SPEEDCHEM_BUILD_DIR` | `~/.cache/speedchem` | where `libksxtb.so` is built |

## Use

```python
from speedchem.gpu.optimize import RelaxTask, relax_many
from speedchem.settings import RelaxSettings

# mol: an RDKit molecule with explicit hydrogens; z its atomic numbers; start in Bohr
out = relax_many([RelaxTask("key", mol, z, start, charge, 0)], RelaxSettings())
```

Each outcome holds the geometry (Bohr), or a `speedchem.reasons.Reason` when the run ended
without one (`OPT_NOT_CONVERGED`, `SCF_NOT_CONVERGED`, `CONNECTIVITY_CHANGED`).

## Tests

```bash
uv sync
uv run pytest -n 8            # the tests marked gpu skip without a CUDA device or nvcc
```

tblite is the reference of every test.

## Licence

LGPL-3.0-or-later (`COPYING.LESSER`, with the GPL it supplements in `COPYING`). The GFN2-xTB
parameters are extracted from tblite, mctc-lib and dftd4: see `NOTICE`.
