"""Convolutional VAE for one-hot encoded SMILES."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


class Sampling(layers.Layer):
    """Reparameterisation trick: ``z = mean + exp(0.5 * log_var) * eps``."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder(max_len: int, vocab_size: int, latent_dim: int) -> models.Model:
    inputs = layers.Input(shape=(max_len, vocab_size))
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(max_len: int, vocab_size: int, latent_dim: int) -> models.Model:
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(max_len * 64, activation="relu")(x)
    x = layers.Reshape((max_len, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", padding="same")(x)
    outputs = layers.Conv1DTranspose(
        vocab_size, 3, activation="softmax", padding="same"
    )(x)
    return models.Model(inputs, outputs, name="decoder")


class VAE(tf.keras.Model):
    """VAE whose KL term is weighted by ``kl_weight``."""

    def __init__(self, encoder, decoder, kl_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        kl = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1.0
        )
        self.add_loss(self.kl_weight * kl)
        return self.decoder(z)


def build_vae(
    max_len: int,
    vocab_size: int,
    latent_dim: int = 2,
    kl_weight: float = 0.1,
    learning_rate: float = 1e-3,
) -> VAE:
    vae = VAE(
        build_encoder(max_len, vocab_size, latent_dim),
        build_decoder(max_len, vocab_size, latent_dim),
        kl_weight=kl_weight,
    )
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
    )
    return vae
