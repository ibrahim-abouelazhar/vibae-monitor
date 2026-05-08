"""
src/train.py — Boucle d'entraînement de l'Autoencoder VibAE (TensorFlow/Keras).

Pipeline :
    load_processed()  →  prépare les tenseurs  →  entraîne le modèle
    →  calcule le seuil sur train  →  sauvegarde modèle + seuil

Usage direct :
    python -m src.train

Import depuis un autre module :
    from src.train import train_autoencoder, load_trained_model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import MODELS_DIR, WINDOW_SIZE
from src.data_loader import load_processed
from src.model import build_conv_autoencoder, build_fc_autoencoder, reconstruction_error

# ---------------------------------------------------------------------------
# Hyperparamètres d'entraînement — modifiables sans toucher à config.py
# ---------------------------------------------------------------------------
EPOCHS         = 50
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
VALIDATION_SPLIT = 0.1     # 10% des fenêtres normales pour la validation interne
PATIENCE       = 10        # EarlyStopping : arrêt si val_loss ne s'améliore plus
THRESHOLD_PERCENTILE = 95  # Percentile de l'erreur train pour le seuil d'anomalie
ARCHITECTURE   = "conv"    # "conv" ou "fc"

# Chemins de sauvegarde
MODEL_PATH     = MODELS_DIR / "autoencoder.keras"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
HISTORY_PATH   = MODELS_DIR / "training_history.json"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _prepare_tensors(
    df_train,
    architecture: str = ARCHITECTURE,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Extrait les colonnes signal du DataFrame et les met en forme pour Keras.

    - Conv1D  : (n_windows, window_size, 1)  — canal explicite requis
    - FC      : (n_windows, window_size)
    """
    window_cols = [f"window_{i}" for i in range(window_size)]
    x = df_train[window_cols].values.astype("float32")   # (n, 1024)

    if architecture == "conv":
        x = x[:, :, np.newaxis]                          # (n, 1024, 1)

    return x


# ---------------------------------------------------------------------------
# Fonction principale d'entraînement
# ---------------------------------------------------------------------------

def train_autoencoder(
    epochs: int         = EPOCHS,
    batch_size: int     = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    architecture: str   = ARCHITECTURE,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    models_dir: Path    = MODELS_DIR,
) -> tuple[keras.Model, float, dict]:
    """Entraîne l'autoencoder sur les fenêtres normales et sauvegarde le résultat.

    Étapes :
        1. Charge df_train via load_processed() — fenêtres normales uniquement
        2. Construit le modèle selon `architecture`
        3. Entraîne avec EarlyStopping sur val_loss
        4. Calcule le seuil d'anomalie (percentile des erreurs train)
        5. Sauvegarde le modèle (.keras), le seuil (JSON) et l'historique (JSON)

    Args:
        epochs:               Nombre maximum d'époques.
        batch_size:           Taille de batch.
        learning_rate:        Taux d'apprentissage Adam.
        architecture:         "conv" (Conv1D) ou "fc" (Fully Connected).
        threshold_percentile: Percentile utilisé pour le seuil (ex : 95).
        models_dir:           Dossier de sauvegarde des artefacts.

    Returns:
        (model, threshold, history_dict)
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Chargement des données
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VibAE — Entraînement Autoencoder (TensorFlow/Keras)")
    print("=" * 60)
    print(f"\n  Architecture      : {architecture.upper()}")
    print(f"  Époques max       : {epochs}")
    print(f"  Batch size        : {batch_size}")
    print(f"  Learning rate     : {learning_rate}")
    print(f"  Seuil (percentile): {threshold_percentile}e\n")

    df_train, df_test, scaler = load_processed()

    # Garde-fou : s'assure que le train ne contient que du normal
    assert (df_train["label"] == "normal").all(), (
        "BUG: df_train contient des défauts !"
    )

    x_train = _prepare_tensors(df_train, architecture=architecture)
    print(f"[Data] Tenseur train : {x_train.shape}")

    # ------------------------------------------------------------------
    # 2. Construction du modèle
    # ------------------------------------------------------------------
    if architecture == "conv":
        model = build_conv_autoencoder(
            window_size=WINDOW_SIZE,
            latent_dim=32,
            learning_rate=learning_rate,
        )
    elif architecture == "fc":
        model = build_fc_autoencoder(
            window_size=WINDOW_SIZE,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Architecture inconnue : '{architecture}'. Choisir 'conv' ou 'fc'.")

    model.summary(line_length=70)

    # ------------------------------------------------------------------
    # 3. Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "autoencoder_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ------------------------------------------------------------------
    # 4. Entraînement
    # (autoencoder : input = output — on reconstruit le signal d'entrée)
    # ------------------------------------------------------------------
    print("\n[Train] Début de l'entraînement...\n")

    history = model.fit(
        x_train,           # input
        x_train,           # target = input (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 5. Calcul du seuil d'anomalie
    # ------------------------------------------------------------------
    print("\n[Threshold] Calcul du seuil d'anomalie sur les erreurs train...")

    train_errors = reconstruction_error(model, x_train, batch_size=batch_size)
    threshold = float(np.percentile(train_errors, threshold_percentile))

    print(f"  Erreur train — min   : {train_errors.min():.6f}")
    print(f"  Erreur train — médiane: {np.median(train_errors):.6f}")
    print(f"  Erreur train — max   : {train_errors.max():.6f}")
    print(f"  Seuil ({threshold_percentile}e pct)  : {threshold:.6f}")

    # ------------------------------------------------------------------
    # 6. Évaluation rapide sur le test (aperçu, pas de métriques finales)
    # ------------------------------------------------------------------
    print("\n[Eval rapide] Aperçu sur df_test...")
    x_test = _prepare_tensors(df_test, architecture=architecture)
    test_errors = reconstruction_error(model, x_test, batch_size=batch_size)
    y_pred = (test_errors > threshold).astype(int)
    y_true = (df_test["label"] != "normal").astype(int).values

    n_anomalies_detected = int(y_pred.sum())
    n_real_faults = int(y_true.sum())
    n_test = len(y_true)

    print(f"  Fenêtres test total      : {n_test:,}")
    print(f"  Défauts réels            : {n_real_faults:,}")
    print(f"  Détectés comme anomalie  : {n_anomalies_detected:,}")

    # Calcul simple précision/rappel (sans sklearn pour éviter l'import)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"  Précision  : {precision:.3f}")
    print(f"  Rappel     : {recall:.3f}")
    print(f"  F1-score   : {f1:.3f}  (seuil brut — affiner dans evaluate.py)")

    # ------------------------------------------------------------------
    # 7. Sauvegarde
    # ------------------------------------------------------------------
    model_path     = models_dir / "autoencoder.keras"
    threshold_path = models_dir / "threshold.json"
    history_path   = models_dir / "training_history.json"

    model.save(str(model_path))

    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "percentile": threshold_percentile,
                "architecture": architecture,
                "train_error_mean": float(train_errors.mean()),
                "train_error_std": float(train_errors.std()),
            },
            f,
            indent=2,
        )

    history_dict = history.history
    # Convertir les valeurs en float Python natif pour la sérialisation JSON
    history_serializable = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    print("\n[Sauvegarde]")
    for p in (model_path, threshold_path, history_path):
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  {p.resolve()}  ({size_mb:.2f} MB)")

    print("\n[OK] Entraînement terminé.")
    print("     → Utilisez evaluate.py pour les métriques complètes (AUC-ROC, F1, matrice de confusion).")

    return model, threshold, history_dict


# ---------------------------------------------------------------------------
# Chargement du modèle entraîné (pour detect.py / evaluate.py)
# ---------------------------------------------------------------------------

def load_trained_model(
    models_dir: Path = MODELS_DIR,
) -> tuple[keras.Model, float]:
    """Recharge l'autoencoder entraîné et le seuil depuis le disque.

    Returns:
        (model, threshold)

    Raises:
        FileNotFoundError si les fichiers sont absents — relancer train.py.
    """
    model_path     = models_dir / "autoencoder.keras"
    threshold_path = models_dir / "threshold.json"

    for p in (model_path, threshold_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {p}\n"
                f"Relancez l'entraînement : python -m src.train"
            )

    model = keras.models.load_model(str(model_path))

    with open(threshold_path) as f:
        meta = json.load(f)

    threshold = meta["threshold"]
    print(f"[load_trained_model] Modèle chargé — seuil d'anomalie : {threshold:.6f}")

    return model, threshold


# ---------------------------------------------------------------------------
# Point d'entrée direct
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_autoencoder()
