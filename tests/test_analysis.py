import pytest

pytest.importorskip("rdkit")

from smiles_vae.analysis import describe, validity_rate


def test_describe_ethanol():
    d = describe("CCO")
    assert d is not None
    assert d.mol_wt == pytest.approx(46.07, abs=0.01)
    assert d.h_donors == 1


def test_invalid_returns_none():
    assert describe("CC((()=") is None
    assert describe("") is None


def test_validity_rate():
    assert validity_rate(["CCO", "CC((()="]) == 0.5
    assert validity_rate([]) == 0.0
