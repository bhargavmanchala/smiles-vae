"""Dataset loading and SMILES encoding utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAD_CHAR = " "

DEFAULT_SMILES = ["CCO", "CC(=O)O", "CCC", "CCN", "CCOCC"]


def load_smiles(path: str | Path) -> list[str]:
    """Read one SMILES per line, ignoring blank lines and ``#`` comments.

    Only the first whitespace-separated token of a line is used, so an
    inline name such as ``CCO ethanol`` is accepted.
    """
    smiles: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            smiles.append(line.split()[0])
    if not smiles:
        raise ValueError(f"No SMILES found in {path}")
    return smiles


def build_vocab(smiles: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build char<->index maps. The pad character always gets index 0."""
    chars = sorted(set("".join(smiles)) - {PAD_CHAR})
    char_to_index = {PAD_CHAR: 0}
    char_to_index.update({c: i + 1 for i, c in enumerate(chars)})
    index_to_char = {i: c for c, i in char_to_index.items()}
    return char_to_index, index_to_char


def one_hot_encode(
    smiles: list[str], max_len: int, char_to_index: dict[str, int]
) -> np.ndarray:
    """Encode SMILES as a ``(n, max_len, vocab)`` one-hot array.

    Positions past the end of a string are set to the pad token, so every
    row sums to one along the vocabulary axis.
    """
    out = np.zeros((len(smiles), max_len, len(char_to_index)), dtype=np.float32)
    out[:, :, char_to_index[PAD_CHAR]] = 1.0
    for i, smi in enumerate(smiles):
        for j, ch in enumerate(smi[:max_len]):
            if ch not in char_to_index:
                raise KeyError(f"Character {ch!r} not in vocabulary")
            out[i, j, :] = 0.0
            out[i, j, char_to_index[ch]] = 1.0
    return out


def decode(one_hot: np.ndarray, index_to_char: dict[int, str]) -> list[str]:
    """Decode ``(n, max_len, vocab)`` probabilities/one-hot back to strings.

    Decoding stops at the first pad token, since anything after it is padding.
    """
    idx = np.argmax(one_hot, axis=-1)
    out = []
    for row in idx:
        text = "".join(index_to_char[int(k)] for k in row)
        out.append(text.split(PAD_CHAR, 1)[0])
    return out
