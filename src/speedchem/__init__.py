"""speedchem: fast geometry optimization for semiempirical quantum chemistry.

- `speedchem.anc`: ANC / RFO optimization (xtb's ANCopt scheme) on any energy-and-gradient
  function, in numpy.
- `speedchem.bonds`: the bond checks of an optimization run (input-bond walls, the dissociation
  guard, the connectivity check), on RDKit molecules.
- `speedchem.gpu`: GFN2-xTB with ALPB water on CUDA (integrals, SCF, gradients) and the batched
  ANC optimizer of many molecules (`speedchem.gpu.optimize.relax_many`). It needs torch and a
  CUDA device; the rest of the package imports neither.
"""

__version__ = "0.1.0"
