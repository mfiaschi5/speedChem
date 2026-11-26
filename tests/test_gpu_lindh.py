"""The vectorized Lindh model Hessian of the GPU optimizer against `speedchem.anc.lindh_hessian`."""

from __future__ import annotations

import numpy as np
import pytest

from speedchem.anc import lindh_hessian
from speedchem.gpu.lindh import lindh_hessian_fast
from tests.test_gpu import CASES

LINEAR = {  # near-linear and linear angles: the bend and torsion cut-offs
    "acetylene_like": (np.array([6, 6, 1, 1]),
                       np.array([[0, 0, 0], [0, 0, 2.27], [0, 0.01, -2.0], [0, 0, 4.27]])),
    "co2": (np.array([8, 6, 8]), np.array([[0, 0, -2.2], [0, 0, 0], [0, 0, 2.2]])),
    "bent_by_1e-4": (np.array([8, 6, 8]), np.array([[0, 0, -2.2], [0, 2e-4, 0], [0, 0, 2.2]])),
}  # fmt: skip


@pytest.mark.parametrize("name", [*CASES, *LINEAR])
def test_vectorized_lindh_hessian_matches_the_loops(name):
    z, xyz = CASES[name][:2] if name in CASES else LINEAR[name]
    a, b = lindh_hessian(z, xyz), lindh_hessian_fast(z, xyz)
    assert np.abs(a - b).max() <= 1e-12 * max(1.0, np.abs(a).max())
