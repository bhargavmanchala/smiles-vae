"""RDKit-based validity checks and descriptor calculation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Descriptors:
    smiles: str
    mol_wt: float
    log_p: float
    h_donors: int
    h_acceptors: int


def describe(smiles: str) -> Descriptors | None:
    """Return descriptors for a valid SMILES, or ``None`` if it cannot be parsed."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors as D
    from rdkit.Chem import Lipinski

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return Descriptors(
        smiles=smiles,
        mol_wt=D.MolWt(mol),
        log_p=D.MolLogP(mol),
        h_donors=Lipinski.NumHDonors(mol),
        h_acceptors=Lipinski.NumHAcceptors(mol),
    )


def validity_rate(smiles: list[str]) -> float:
    """Fraction of strings that RDKit parses as molecules."""
    if not smiles:
        return 0.0
    return sum(describe(s) is not None for s in smiles) / len(smiles)
