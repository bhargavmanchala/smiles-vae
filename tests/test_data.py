import numpy as np
import pytest

from smiles_vae.data import PAD_CHAR, build_vocab, decode, load_smiles, one_hot_encode

SMILES = ["CCO", "CC(=O)O", "CCC"]


def test_vocab_pad_is_zero():
    c2i, i2c = build_vocab(SMILES)
    assert c2i[PAD_CHAR] == 0
    assert all(i2c[i] == c for c, i in c2i.items())


def test_one_hot_rows_sum_to_one():
    c2i, _ = build_vocab(SMILES)
    x = one_hot_encode(SMILES, 7, c2i)
    assert x.shape == (3, 7, len(c2i))
    assert np.allclose(x.sum(axis=-1), 1.0)


def test_roundtrip():
    c2i, i2c = build_vocab(SMILES)
    x = one_hot_encode(SMILES, 7, c2i)
    assert decode(x, i2c) == SMILES


def test_unknown_char_raises():
    c2i, _ = build_vocab(SMILES)
    with pytest.raises(KeyError):
        one_hot_encode(["CCZ"], 3, c2i)


def test_load_smiles(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# c\nCCO ethanol\n\nCCC\n", encoding="utf-8")
    assert load_smiles(f) == ["CCO", "CCC"]


def test_load_smiles_empty(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# only comment\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_smiles(f)


def test_decode_stops_at_first_pad():
    c2i, i2c = build_vocab(SMILES)
    x = one_hot_encode(["CC"], 4, c2i)
    x[0, 3] = one_hot_encode(["C"], 4, c2i)[0, 0]  # stray char after padding
    assert decode(x, i2c) == ["CC"]
