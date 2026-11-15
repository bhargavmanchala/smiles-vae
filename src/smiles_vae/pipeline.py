"""High-level train and generate helpers shared by the CLI and notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import build_vocab, decode, one_hot_encode
from .model import VAE, build_vae


@dataclass
class Trained:
    vae: VAE
    history: dict
    char_to_index: dict[str, int]
    index_to_char: dict[int, str]
    max_len: int
    latent_dim: int


def set_seed(seed: int) -> None:
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(
    smiles: list[str],
    epochs: int = 25,
    batch_size: int = 2,
    latent_dim: int = 2,
    kl_weight: float = 0.1,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    seed: int = 42,
    verbose: int = 0,
) -> Trained:
    set_seed(seed)
    char_to_index, index_to_char = build_vocab(smiles)
    max_len = max(len(s) for s in smiles)
    x = one_hot_encode(smiles, max_len, char_to_index)
    vae = build_vae(max_len, len(char_to_index), latent_dim, kl_weight, learning_rate)
    hist = vae.fit(
        x,
        x,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return Trained(vae, hist.history, char_to_index, index_to_char, max_len, latent_dim)


def generate(t: Trained, n: int = 10, seed: int | None = None) -> list[str]:
    """Sample ``n`` latent vectors from N(0, I) and decode them."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, t.latent_dim)).astype(np.float32)
    probs = t.vae.decoder.predict(z, verbose=0)
    return decode(probs, t.index_to_char)
