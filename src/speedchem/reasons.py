"""Why an optimization ended without a geometry.

The values are the reason codes of rhoMech's QM records (`ksmech.qm.records.ReasonCode`, a
StrEnum too), so an outcome compares equal to the caller's code of the same name; compare by
value (`==`), never by identity, across the two.
"""

from __future__ import annotations

from enum import StrEnum


class Reason(StrEnum):
    CONNECTIVITY_CHANGED = "CONNECTIVITY_CHANGED"  # an input bond broke
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    OPT_NOT_CONVERGED = "OPT_NOT_CONVERGED"


class ConnectivityChanged(Exception):
    """An input bond stretched beyond its limit (`speedchem.bonds`); `detail` says which."""

    reason = Reason.CONNECTIVITY_CHANGED

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail
