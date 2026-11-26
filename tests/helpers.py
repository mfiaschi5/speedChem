"""What the tests need from outside speedchem: tblite as the reference, and starts embedded by
RDKit (ETKDGv3 with a seed, the lowest of ten MMFF-relaxed conformers, as rhoMech embeds them)."""

from __future__ import annotations

from functools import cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tblite.interface import Calculator

_ANGULAR = "spdfg"


def new_calculator(z, xyz_bohr, charge, unpaired, solvent=None, save_integrals=False,
                   scf_maxiter=250) -> Calculator:  # fmt: skip
    """tblite's GFN2-xTB, quiet, with ALPB `solvent` if given."""
    calc = Calculator("GFN2-xTB", np.asarray(z), np.asarray(xyz_bohr), charge=float(charge),
                      uhf=int(unpaired))  # fmt: skip
    calc.set("verbosity", 0)
    calc.set("max-iter", int(scf_maxiter))
    if solvent:
        calc.add("alpb-solvation", solvent)
    if save_integrals:
        calc.set("save-integrals", 1)
    return calc


@cache
def gfn2_shells() -> dict[int, tuple[tuple[int, int, float, int], ...]]:
    """Atomic number -> ((n, l, Slater exponent, primitives), ...) per shell in tblite's order,
    from the GFN2-xTB parametrization tblite exports."""
    from tblite import library

    param = library.new_param()
    library.export_gfn2_param(param)
    table = library.new_table()
    library.dump_param(param, table)
    elements = library.table_to_dict(table)["element"]
    pt = Chem.GetPeriodicTable()
    out = {}
    for symbol, rec in elements.items():
        shells = zip(rec["shells"], rec["slater"], rec["ngauss"], strict=True)
        out[pt.GetAtomicNumber(symbol)] = tuple(
            (int(name[:-1]), _ANGULAR.index(name[-1]), float(zeta), int(ng))
            for name, zeta, ng in shells
        )
    return out


def element_ao_l(z: int) -> tuple[int, ...]:
    """Angular momentum of every AO of element `z`, in tblite's order."""
    return tuple(ell for _, ell, _, _ in gfn2_shells()[int(z)] for _ in range(2 * ell + 1))


def prepare_mol(smiles: str) -> Chem.Mol:
    """The molecule of a SMILES with explicit hydrogens (heavy atoms in SMILES order, then H)."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def embed(mol: Chem.Mol, n_conformers: int = 1, seed: int = 0) -> list[int]:
    """Embed `mol` in place: the `n_conformers` lowest of max(3n, 10) ETKDGv3 conformers after
    MMFF relaxation, renumbered from 0 (lowest first)."""
    params = AllChem.ETKDGv3()
    params.randomSeed, params.numThreads = seed, 1
    cids = list(AllChem.EmbedMultipleConfs(mol, max(3 * n_conformers, 10), params))
    energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)]
    keep = [cids[i] for i in np.argsort(energies)[:n_conformers]]
    confs = [Chem.Conformer(mol.GetConformer(c)) for c in keep]
    mol.RemoveAllConformers()
    for k, conf in enumerate(confs):
        conf.SetId(k)
        mol.AddConformer(conf, assignId=False)
    return list(range(len(confs)))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of two (n, 3) coordinate sets after optimal rigid superposition."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    s[-1] *= np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(max((a**2).sum() + (b**2).sum() - 2 * s.sum(), 0.0) / len(a)))
