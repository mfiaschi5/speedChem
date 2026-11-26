"""The settings of an optimization run (`speedchem.gpu.optimize.relax_many`).

Any object with these attributes will do: rhoMech passes its `QMConfig`, whose defaults are
these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelaxSettings:
    fmax: float = 3e-4  # converged when max |gradient| (walls included) is below, Hartree/Bohr
    maxiter: int = 500  # energy evaluations of one run (ANC: steps taken back included)
    scf_maxiter: int = 250  # SCF cycles per energy (tblite's default)
    conn_cov_factor: float = 1.3  # an input bond beyond this x its covalent-radius sum is broken
    restrain: bool = True  # after a broken bond, a run with walls on the input bonds
    wall_factor: float = 1.2  # the walls start at this x the covalent-radius sum
    wall_k: float = 1.0  # Hartree / Bohr^2
