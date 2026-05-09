"""
Chargement et découpage en fenêtres des signaux CWRU.

Pipeline :  fichier .mat  →  signal 1D  →  fenêtres glissantes  →  DataFrame
           → split (normal/défauts)  →  normalisation  →  sauvegarde parquet
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DEV_MODE,
    DEV_SUBSET_SIZE,
    MODELS_DIR,
    OVERLAP,
    RANDOM_SEED,
    SAMPLING_RATE,
    TRAIN_RATIO,
    WINDOW_SIZE,
)

# Ordre de priorité utilisé en DEV_MODE pour garantir la diversité des labels
_LABEL_PRIORITY = ["normal", "IR", "OR", "Ball"]


# ---------------------------------------------------------------------------
# Fonction 1 : lecture d'un fichier .mat
# ---------------------------------------------------------------------------


def load_mat_file(filepath: Path) -> tuple[np.ndarray, int]:
    """Lit un fichier .mat CWRU et retourne (signal_1D, sampling_rate).

    Recherche la clé Drive End (pattern *_DE_time) et aplatit le tableau
    en 1D si nécessaire. Suppose une fréquence d'échantillonnage de 12 kHz.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    mat_data = scipy.io.loadmat(str(filepath))

    # Filtre les métadonnées internes scipy (__header__, __version__, etc.)
    de_keys = [k for k in mat_data if not k.startswith("__") and k.endswith("_DE_time")]

    if not de_keys:
        clés_disponibles = [k for k in mat_data if not k.startswith("__")]
        raise ValueError(
            f"Aucune clé '*_DE_time' trouvée dans {filepath.name}.\n"
            f"Clés disponibles : {clés_disponibles}"
        )

    signal = mat_data[de_keys[0]]

    # (N, 1) → (N,)  ou déjà (N,)
    if signal.ndim > 1:
        signal = signal.flatten()

    return signal.astype(np.float32), SAMPLING_RATE


# ---------------------------------------------------------------------------
# Fonction 2 : détection du label depuis le nom de fichier
# ---------------------------------------------------------------------------


def detect_label(filename: str) -> str:
    """Détecte le label de défaut depuis le préfixe du nom de fichier.

    Règles : Normal→'normal', IR→'IR', OR→'OR', B<chiffre>→'Ball', sinon 'unknown'.
    """
    stem = Path(filename).stem  # supprime l'extension si présente

    if stem.upper().startswith("NORMAL"):
        return "normal"
    if stem.upper().startswith("IR"):
        return "IR"
    if stem.upper().startswith("OR"):
        return "OR"
    # 'B' suivi d'un chiffre → Ball fault (ex: B007, B014, B021)
    # Garde-fou : évite de matcher un nom commençant par 'Ball' ou autre mot
    if stem.upper().startswith("B") and len(stem) > 1 and stem[1].isdigit():
        return "Ball"

    return "unknown"


# ---------------------------------------------------------------------------
# Fonction 3 : découpage en fenêtres glissantes (vectorisé)
# ---------------------------------------------------------------------------


def sliding_window(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """Découpe un signal 1D en fenêtres avec recouvrement.

    Retourne un array 2D de shape (n_windows, window_size).
    La dernière fenêtre incomplète est ignorée.
    """
    step = int(window_size * (1.0 - overlap))  # ex : 1024 * 0.5 = 512

    # sliding_window_view retourne une vue sans copie de mémoire : shape (L, window_size)
    # L = nombre de positions de départ possibles (chaque point du signal)
    view = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_size)

    # On ne garde que les positions alignées sur le pas `step`
    windows = view[::step]  # shape (n_windows, window_size)

    return windows.copy()  # .copy() pour libérer la vue sur le signal original


# ---------------------------------------------------------------------------
# Fonction 4 : chargement de l'ensemble du dataset
# ---------------------------------------------------------------------------


def load_all_signals(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Charge tous les .mat, applique sliding window, retourne un DataFrame.

    Colonnes : window_0 … window_<WINDOW_SIZE-1>, label, source_file.
    En DEV_MODE, ne charge que DEV_SUBSET_SIZE fichiers avec diversité maximale.
    """
    mat_files = sorted(data_dir.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(
            f"Aucun fichier .mat dans {data_dir}.\n"
            "Vérifiez que data/raw/ contient bien les fichiers CWRU."
        )

    # --- Sélection DEV_MODE ---
    if DEV_MODE:
        mat_files = _select_dev_subset(mat_files, DEV_SUBSET_SIZE)
        print(f"[DEV_MODE] {len(mat_files)} fichiers sélectionnés : {[f.name for f in mat_files]}")

    # --- Boucle de chargement ---
    colonnes_signal = [f"window_{i}" for i in range(WINDOW_SIZE)]
    frames: list[pd.DataFrame] = []

    for mat_path in tqdm(mat_files, desc="Chargement des fichiers .mat", unit="fichier"):
        try:
            signal, _ = load_mat_file(mat_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [AVERTISSEMENT] {mat_path.name} ignoré : {exc}")
            continue

        label = detect_label(mat_path.name)
        fenetres = sliding_window(signal)  # (n_windows, WINDOW_SIZE)

        df_fichier = pd.DataFrame(fenetres, columns=colonnes_signal)
        df_fichier["label"] = label
        df_fichier["source_file"] = mat_path.name
        frames.append(df_fichier)

    if not frames:
        raise RuntimeError("Aucune fenêtre chargée — vérifiez le contenu de data/raw/.")

    df = pd.concat(frames, ignore_index=True)

    # --- Résumé ---
    nb_fichiers = df["source_file"].nunique()
    nb_fenetres = len(df)
    repartition = df["label"].value_counts()

    print(f"\n{'─' * 50}")
    print(f"  Fichiers chargés   : {nb_fichiers}")
    print(f"  Fenêtres totales   : {nb_fenetres:,}")
    print(f"  Taille fenêtre     : {WINDOW_SIZE} pts  |  overlap : {OVERLAP:.0%}")
    print(f"  Répartition par label :")
    for label_name, count in repartition.items():
        print(f"    {label_name:<10}  {count:>6,} fenêtres")
    print(f"{'─' * 50}\n")

    return df


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _select_dev_subset(mat_files: list[Path], n: int) -> list[Path]:
    """Sélectionne n fichiers en priorisant la diversité des labels.

    Prend 1 fichier par label présent (dans l'ordre _LABEL_PRIORITY),
    puis complète jusqu'à n si besoin.
    """
    # Groupe les fichiers par label
    par_label: dict[str, list[Path]] = {}
    for f in mat_files:
        lbl = detect_label(f.name)
        par_label.setdefault(lbl, []).append(f)

    selected: list[Path] = []

    # 1 fichier par catégorie prioritaire d'abord
    for lbl in _LABEL_PRIORITY:
        if lbl in par_label and len(selected) < n:
            selected.append(par_label[lbl][0])

    # Complète si n > nombre de catégories
    if len(selected) < n:
        for f in mat_files:
            if f not in selected:
                selected.append(f)
            if len(selected) >= n:
                break

    return selected[:n]


# ---------------------------------------------------------------------------
# Fonction 5 : split pour la détection d'anomalies non supervisée
# ---------------------------------------------------------------------------


def split_anomaly_detection(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split spécifique à la détection d'anomalies non supervisée.

    Train : train_ratio des fenêtres normales uniquement (l'autoencoder
    apprend exclusivement la distribution saine).
    Test  : les normales restantes + 100% des défauts, mélangés.
    """
    df_normal = df[df["label"] == "normal"].copy()
    df_faults = df[df["label"] != "normal"].copy()

    # Mélange reproductible des normales
    df_normal = df_normal.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(len(df_normal) * train_ratio)
    df_normal_train = df_normal.iloc[:n_train]
    df_normal_test  = df_normal.iloc[n_train:]

    df_train = df_normal_train.reset_index(drop=True)

    # Test = normales restantes + tous les défauts, mélangés
    df_test = (
        pd.concat([df_normal_test, df_faults], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # Garde-fou critique : le train ne doit JAMAIS contenir de défauts
    assert (df_train["label"] == "normal").all(), (
        "BUG: df_train contient des défauts !"
    )

    # Récap
    print(f"\n[Split] Train : {len(df_train):,} fenêtres normales")
    test_counts = df_test["label"].value_counts()
    details = "  ".join(f"{lbl}={cnt:,}" for lbl, cnt in test_counts.items())
    print(f"[Split] Test  : {len(df_test):,} fenêtres total  ({details})")

    return df_train, df_test


# ---------------------------------------------------------------------------
# Fonction 6 : normalisation MinMaxScaler (anti data-leakage)
# ---------------------------------------------------------------------------


def normalize(
    df: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Normalise les colonnes window_* avec MinMaxScaler dans [0, 1].

    Si scaler est None : fit + transform (cas train).
    Sinon : transform uniquement (cas test — évite le data leakage).
    """
    colonnes_signal = [col for col in df.columns if col.startswith("window_")]
    colonnes_meta   = [col for col in df.columns if not col.startswith("window_")]

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        valeurs_norm = scaler.fit_transform(df[colonnes_signal])
    else:
        valeurs_norm = scaler.transform(df[colonnes_signal])

    df_norm = pd.DataFrame(valeurs_norm, columns=colonnes_signal, index=df.index)
    # Réattache les colonnes méta (label, source_file) sans les toucher
    df_norm[colonnes_meta] = df[colonnes_meta].values

    return df_norm, scaler


# ---------------------------------------------------------------------------
# Fonction 7 : sauvegarde du pipeline préparé
# ---------------------------------------------------------------------------


def save_processed(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: MinMaxScaler,
    output_dir: Path = DATA_PROCESSED,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Sauvegarde df_train, df_test (parquet) et scaler (joblib) sur disque."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    chemin_train  = output_dir  / "df_train.parquet"
    chemin_test   = output_dir  / "df_test.parquet"
    chemin_scaler = models_dir  / "scaler.pkl"

    df_train.to_parquet(chemin_train,  index=False)
    df_test.to_parquet(chemin_test,   index=False)
    joblib.dump(scaler, chemin_scaler)

    print("\n[Sauvegarde]")
    for chemin in (chemin_train, chemin_test, chemin_scaler):
        taille_mb = chemin.stat().st_size / (1024 ** 2)
        print(f"  {chemin.resolve()}  ({taille_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Fonction 8 : rechargement du pipeline (pour P3 — entraînement)
# ---------------------------------------------------------------------------


def load_processed(
    processed_dir: Path = DATA_PROCESSED,
    models_dir: Path = MODELS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Recharge df_train, df_test et scaler depuis disque.

    Lève FileNotFoundError si un fichier est absent — relancer
    `python -m src.data_loader` pour régénérer les fichiers.
    """
    chemins = {
        "df_train" : processed_dir / "df_train.parquet",
        "df_test"  : processed_dir / "df_test.parquet",
        "scaler"   : models_dir    / "scaler.pkl",
    }

    for nom, chemin in chemins.items():
        if not chemin.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {chemin}\n"
                f"Relancez le pipeline : python -m src.data_loader"
            )

    df_train = pd.read_parquet(chemins["df_train"])
    df_test  = pd.read_parquet(chemins["df_test"])
    scaler   = joblib.load(chemins["scaler"])

    print(f"[load_processed] df_train={df_train.shape}  df_test={df_test.shape}  scaler chargé")
    return df_train, df_test, scaler


# ---------------------------------------------------------------------------
# Point d'entrée direct — pipeline complet bout en bout
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline data_loader vibae-monitor")
    print("=" * 60)

    # 1. Chargement
    df = load_all_signals()

    # 2. Split AVANT normalisation (anti data-leakage)
    df_train, df_test = split_anomaly_detection(df)

    # 3. Normalisation : fit sur train, transform sur test
    df_train, scaler = normalize(df_train)
    df_test, _       = normalize(df_test, scaler=scaler)

    # 4. Sauvegarde
    save_processed(df_train, df_test, scaler)

    print("\n[OK] Pipeline termine. Les fichiers suivants sont prets :")
    print("  data/processed/df_train.parquet")
    print("  data/processed/df_test.parquet")
    print("  models/scaler.pkl")
    print("\nP3 (ML) peut maintenant utiliser load_processed() pour entrainer l'autoencoder.")
