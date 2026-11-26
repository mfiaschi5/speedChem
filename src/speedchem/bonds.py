"""The bond checks of an optimization run, on RDKit molecules with explicit hydrogens.

- `bond_walls`: flat-bottom harmonic walls on the input bonds, for a run retried after a bond
  broke;
- `DissociationGuard`: stops a free run once an input bond has stayed stretched;
- `check_connectivity`: whether an end geometry kept every input bond.

Bonds to Li, Na, Mg and K are not tested: their contacts with the rest of the molecule are
electrostatic, and covalent-radius ratios mean little for them. A broken bond raises
`speedchem.reasons.ConnectivityChanged`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from speedchem.reasons import ConnectivityChanged
from speedchem.units import BOHR_ANG

SKIP_BOND_ELEMENTS = frozenset({3, 11, 12, 19})  # Li, Na, Mg, K
DISSOCIATED_PATIENCE = 5  # the guard stops a free run after this many steps with a bond broken


@dataclass(frozen=True)
class WallRestraint:
    """Flat-bottom harmonic walls E = k * max(0, d - d0)^2 on selected atom pairs."""

    pairs: np.ndarray  # (m, 2) atom indices
    d0_bohr: np.ndarray  # (m,) wall onset distances
    k: float  # Hartree / Bohr^2

    def energy_gradient(self, xyz_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(xyz_bohr)
        if not len(self.pairs):
            return 0.0, grad
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        vec = xyz_bohr[i] - xyz_bohr[j]
        dist = np.linalg.norm(vec, axis=1)
        over = np.maximum(dist - self.d0_bohr, 0.0)
        pull = (2.0 * self.k * over / np.maximum(dist, 1e-12))[:, None] * vec
        np.add.at(grad, i, pull)
        np.add.at(grad, j, -pull)
        return float(self.k * (over**2).sum()), grad


def bond_walls(mol: Chem.Mol, wall_factor: float, k: float) -> WallRestraint:
    """Walls on every input bond (Li, Na, Mg, K bonds excluded) at `wall_factor` x radii sum."""
    table = Chem.GetPeriodicTable()
    pairs, d0 = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() in SKIP_BOND_ELEMENTS or b.GetAtomicNum() in SKIP_BOND_ELEMENTS:
            continue
        pairs.append((a.GetIdx(), b.GetIdx()))
        radii = table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(b.GetAtomicNum())
        d0.append(wall_factor * radii / BOHR_ANG)
    return WallRestraint(np.array(pairs, dtype=int).reshape(-1, 2), np.array(d0), k)


class DissociationGuard:
    """Called with every trial geometry (Bohr): raises `ConnectivityChanged` once an input bond
    (Li, Na, Mg and K bonds excepted) has been longer than `factor` x its covalent-radius sum at
    `patience` trial geometries in a row. In rhoMech's benchmark of 2026-11-09 no free run that
    ended with its bonds intact ever passed 1.3x on the way (99 runs), while a bond that broke
    passed it within a few dozen steps and stayed: the free run need not go on to its end."""

    def __init__(self, mol: Chem.Mol, factor: float, patience: int = DISSOCIATED_PATIENCE) -> None:
        table = Chem.GetPeriodicTable()
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        pairs = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in mol.GetBonds()
            if z[b.GetBeginAtomIdx()] not in SKIP_BOND_ELEMENTS
            and z[b.GetEndAtomIdx()] not in SKIP_BOND_ELEMENTS
        ]
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        radius = table.GetRcovalent
        self.rcov = np.array([radius(z[i]) + radius(z[j]) for i, j in pairs])
        self.factor, self.patience, self.streak = factor, patience, 0

    def __call__(self, xyz_bohr: np.ndarray) -> None:
        if not len(self.pairs):
            return
        d = np.linalg.norm(xyz_bohr[self.pairs[:, 0]] - xyz_bohr[self.pairs[:, 1]], axis=1)
        ratio = d * BOHR_ANG / self.rcov
        k = int(ratio.argmax())
        self.streak = self.streak + 1 if ratio[k] > self.factor else 0
        if self.streak >= self.patience:
            i, j = self.pairs[k]
            raise ConnectivityChanged(
                f"bond ({i},{j}) at {ratio[k]:.2f}x, beyond {self.factor}x for "
                f"{self.patience} steps of the optimization"
            )


def bond_stretch_ratios(mol: Chem.Mol, xyz_ang: np.ndarray) -> list[tuple[int, int, float]]:
    """(i, j, d_ij / (r_cov_i + r_cov_j)) for every input bond, skipping Li, Na, Mg and K bonds."""
    table = Chem.GetPeriodicTable()
    out = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        zi, zj = bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()
        if zi in SKIP_BOND_ELEMENTS or zj in SKIP_BOND_ELEMENTS:
            continue
        d = float(np.linalg.norm(xyz_ang[i] - xyz_ang[j]))
        out.append((i, j, d / (table.GetRcovalent(zi) + table.GetRcovalent(zj))))
    return out


def check_connectivity(mol: Chem.Mol, xyz_ang: np.ndarray, cov_factor: float) -> None:
    """Raise `ConnectivityChanged` if an input bond stretched beyond `cov_factor` x the sum of
    covalent radii. New close contacts are ignored."""
    broken = [(i, j, r) for i, j, r in bond_stretch_ratios(mol, xyz_ang) if r > cov_factor]
    if broken:
        shown = ", ".join(f"({i},{j}) {r:.2f}x" for i, j, r in broken[:6])
        raise ConnectivityChanged(f"{len(broken)} bond(s) beyond {cov_factor}x: {shown}")
