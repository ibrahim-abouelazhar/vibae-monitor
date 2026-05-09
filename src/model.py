"""
src/model.py — Architecture Autoencoder 1D pour vibae-monitor.

Deux architectures disponibles :
  - VibAEAutoencoder     : Conv1D (recommandé — capte les patterns temporels)
  - VibAEAutoencoderFC   : Fully Connected (plus simple, bonne baseline)

Usage :
    from src.model import build_conv_autoencoder, build_fc_autoencoder

    model = build_conv_autoencoder(window_size=1024)
    model.summary()
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Architecture 1 : Autoencoder Conv1D (recommandé)
# ---------------------------------------------------------------------------

def build_conv_autoencoder(
    window_size: int = 1024,
    latent_dim: int = 32,
    filters: tuple[int, ...] = (32, 16, 8),
    kernel_size: int = 7,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Construit et compile un Autoencoder convolutif 1D.

    Architecture :
        Encoder : Conv1D(32) → Conv1D(16) → Conv1D(8) → Dense(latent_dim)
        Decoder : Dense → Reshape → Conv1DTranspose(8) → Conv1DTranspose(16)
                  → Conv1DTranspose(32) → Conv1D(1) sortie

    Args:
        window_size:    Nombre de points par fenêtre (= WINDOW_SIZE dans config).
        latent_dim:     Dimension de l'espace latent.
        filters:        Nombre de filtres pour chaque couche Conv de l'encodeur.
        kernel_size:    Taille du noyau de convolution (impair recommandé).
        learning_rate:  Taux d'apprentissage pour Adam.

    Returns:
        keras.Model compilé (loss=MSE, optimizer=Adam).
    """
    # ---- Encodeur ----
    inp = keras.Input(shape=(window_size, 1), name="input_window")

    x = inp
    for i, f in enumerate(filters):
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"enc_conv_{i+1}",
        )(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same", name=f"enc_pool_{i+1}")(x)

    # Aplatir pour l'espace latent
    shape_before_flatten = x.shape[1:]           # ex : (128, 8) après 3 poolings sur 1024
    x = layers.Flatten(name="flatten")(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # ---- Décodeur ----
    # Remonter vers la forme avant flatten
    n_after_pool = shape_before_flatten[0]        # ex : 128
    last_filters = shape_before_flatten[1]        # ex : 8

    x = layers.Dense(n_after_pool * last_filters, activation="relu", name="dec_dense")(encoded)
    x = layers.Reshape((n_after_pool, last_filters), name="dec_reshape")(x)

    for i, f in enumerate(reversed(filters)):
        x = layers.Conv1DTranspose(
            filters=f,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_convT_{i+1}",
        )(x)

    # Couche de sortie : 1 canal, activation sigmoid pour signal normalisé [0,1]
    decoded = layers.Conv1D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        name="output_window",
    )(x)

    # Rogner ou paddér si la taille ne correspond pas exactement à window_size
    decoded = layers.Cropping1D(
        cropping=_compute_cropping(decoded.shape[1], window_size),
        name="crop_to_window",
    )(decoded)

    model = keras.Model(inputs=inp, outputs=decoded, name="VibAEConvAutoencoder")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------------
# Architecture 2 : Autoencoder Fully Connected (baseline)
# ---------------------------------------------------------------------------

def build_fc_autoencoder(
    window_size: int = 1024,
    hidden_dims: tuple[int, ...] = (512, 128, 32),
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Construit un Autoencoder fully-connected (MLP symétrique).

    Architecture :
        Encoder : Dense(512) → Dense(128) → Dense(32) [latent]
        Decoder : Dense(128) → Dense(512) → Dense(window_size)

    Args:
        window_size:    Taille de la fenêtre d'entrée.
        hidden_dims:    Dimensions des couches cachées (ordre encodeur).
        learning_rate:  Taux d'apprentissage Adam.
        dropout_rate:   Dropout appliqué après chaque couche cachée.

    Returns:
        keras.Model compilé.
    """
    inp = keras.Input(shape=(window_size,), name="input_window")

    # Encodeur
    x = inp
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation="relu", name=f"enc_dense_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"enc_dropout_{i+1}")(x)

    # Décodeur (symétrique, sans la dernière dim qui est le latent)
    for i, dim in enumerate(reversed(hidden_dims[:-1])):
        x = layers.Dense(dim, activation="relu", name=f"dec_dense_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dec_dropout_{i+1}")(x)

    # Couche de sortie
    decoded = layers.Dense(window_size, activation="sigmoid", name="output_window")(x)

    model = keras.Model(inputs=inp, outputs=decoded, name="VibAEFCAutoencoder")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _compute_cropping(current_size: int | None, target_size: int) -> tuple[int, int]:
    """Calcule le rogning symétrique pour ajuster la taille à target_size."""
    if current_size is None or current_size <= target_size:
        return (0, 0)
    diff = current_size - target_size
    left = diff // 2
    right = diff - left
    return (left, right)


def reconstruction_error(
    model: keras.Model,
    x: "np.ndarray",
    batch_size: int = 256,
) -> "np.ndarray":
    """Calcule l'erreur de reconstruction MSE par fenêtre.

    Args:
        model:       Autoencoder entraîné.
        x:           Array (n_windows, window_size) ou (n_windows, window_size, 1).
        batch_size:  Taille de batch pour la prédiction.

    Returns:
        np.ndarray de shape (n_windows,) — score d'anomalie par fenêtre.
    """
    import numpy as np

    x_pred = model.predict(x, batch_size=batch_size, verbose=0)

    # Aplatir si Conv1D (n, L, 1) → (n, L)
    if x_pred.ndim == 3:
        x_pred = x_pred.squeeze(-1)
    x_flat = x.squeeze(-1) if x.ndim == 3 else x

    # MSE par fenêtre
    errors = np.mean((x_flat - x_pred) ** 2, axis=1)
    return errors


# ---------------------------------------------------------------------------
# Test rapide (python -m src.model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Test de l'architecture VibAE (TensorFlow/Keras)")
    print("=" * 60)

    # --- Conv Autoencoder ---
    print("\n[1] Conv1D Autoencoder")
    model_conv = build_conv_autoencoder(window_size=1024, latent_dim=32)
    model_conv.summary()

    x_dummy = np.random.rand(8, 1024, 1).astype("float32")
    x_out   = model_conv.predict(x_dummy, verbose=0)
    print(f"  Input  : {x_dummy.shape}")
    print(f"  Output : {x_out.shape}  (doit être (8, 1024, 1))")
    assert x_out.shape == (8, 1024, 1), "Mauvaise forme de sortie Conv AE !"

    errors = reconstruction_error(model_conv, x_dummy)
    print(f"  Erreurs de reconstruction : min={errors.min():.4f}  max={errors.max():.4f}")

    # --- FC Autoencoder ---
    print("\n[2] Fully Connected Autoencoder")
    model_fc = build_fc_autoencoder(window_size=1024)
    model_fc.summary()

    x_flat   = np.random.rand(8, 1024).astype("float32")
    x_out_fc = model_fc.predict(x_flat, verbose=0)
    print(f"  Input  : {x_flat.shape}")
    print(f"  Output : {x_out_fc.shape}  (doit être (8, 1024))")
    assert x_out_fc.shape == (8, 1024), "Mauvaise forme de sortie FC AE !"

    print("\n[OK] Tous les tests passent.")
