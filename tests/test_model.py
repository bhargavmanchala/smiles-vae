import pytest

pytest.importorskip("tensorflow")

from smiles_vae.data import DEFAULT_SMILES
from smiles_vae.pipeline import generate, train


def test_train_and_generate_shapes():
    t = train(DEFAULT_SMILES, epochs=2, seed=0)
    assert len(t.history["loss"]) == 2
    out = generate(t, n=4, seed=0)
    assert len(out) == 4
    assert all(len(s) <= t.max_len for s in out)
