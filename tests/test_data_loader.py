"""
Tests unitaires pour src/data_loader.py.

Ces tests sont entièrement isolés des fichiers .mat réels.
Tous les signaux et DataFrames utilisés sont synthétiques.

Lancer avec :
    pytest tests/test_data_loader.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

# Assure que le projet est dans sys.path quelle que soit l'origine de l'appel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    detect_label,
    normalize,
    sliding_window,
    split_anomaly_detection,
)

# ---------------------------------------------------------------------------
# Fixtures partagées
# ---------------------------------------------------------------------------

N_WINDOW_COLS = 8  # On utilise 8 colonnes window_* dans les tests (pas 1024)
                   # Comportement identique, empreinte mémoire négligeable.


def _make_window_cols(n: int) -> list[str]:
    return [f"window_{i}" for i in range(n)]


@pytest.fixture
def df_multi_label() -> pd.DataFrame:
    """DataFrame factice avec 100 normales + 50 IR + 30 OR + 20 Ball."""
    rng = np.random.default_rng(0)
    cols = _make_window_cols(N_WINDOW_COLS)

    frames = []
    for label, count in [("normal", 100), ("IR", 50), ("OR", 30), ("Ball", 20)]:
        data = rng.random((count, N_WINDOW_COLS)).astype(np.float32)
        df_part = pd.DataFrame(data, columns=cols)
        df_part["label"] = label
        df_part["source_file"] = f"{label}_0.mat"
        frames.append(df_part)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def df_normal_only(df_multi_label: pd.DataFrame) -> pd.DataFrame:
    """Sous-ensemble ne contenant que les lignes normales."""
    return df_multi_label[df_multi_label["label"] == "normal"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests sliding_window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_shape_standard(self):
        """5000 points, window=1024, overlap=0.5 → 8 fenêtres de 1024."""
        signal = np.zeros(5000, dtype=np.float32)
        windows = sliding_window(signal, window_size=1024, overlap=0.5)
        # step = 512  →  n = (5000 - 1024) // 512 + 1 = 8
        assert windows.shape == (8, 1024)

    def test_shape_no_overlap(self):
        """4096 points, window=1024, overlap=0 → 4 fenêtres exactes."""
        signal = np.zeros(4096, dtype=np.float32)
        windows = sliding_window(signal, window_size=1024, overlap=0.0)
        assert windows.shape == (4, 1024)

    def test_content_first_window(self):
        """La première fenêtre doit contenir les points [0, 1024)."""
        signal = np.arange(2048, dtype=np.float32)
        windows = sliding_window(signal, window_size=1024, overlap=0.5)
        np.testing.assert_array_equal(windows[0], np.arange(1024, dtype=np.float32))

    def test_content_second_window_offset(self):
        """Avec overlap=0.5, la 2e fenêtre commence à l'index step=512."""
        signal = np.arange(2048, dtype=np.float32)
        windows = sliding_window(signal, window_size=1024, overlap=0.5)
        assert windows[1][0] == 512.0

    def test_returns_copy_not_view(self):
        """Le résultat est une copie indépendante, pas une vue sur le signal."""
        signal = np.ones(5000, dtype=np.float32)
        windows = sliding_window(signal, window_size=1024, overlap=0.5)
        assert windows.base is None

    def test_short_signal_raises(self):
        """Signal plus court que window_size → ValueError (numpy sliding_window_view).

        Note : l'implémentation actuelle ne gère pas les signaux courts.
        Ce test documente le comportement réel.
        """
        signal = np.zeros(500, dtype=np.float32)
        with pytest.raises(ValueError):
            sliding_window(signal, window_size=1024, overlap=0.5)


# ---------------------------------------------------------------------------
# Tests detect_label
# ---------------------------------------------------------------------------


class TestDetectLabel:
    @pytest.mark.parametrize("filename,expected", [
        ("Normal_0.mat",    "normal"),   # préfixe Normal standard
        ("normal_test.mat", "normal"),   # insensible à la casse (stem.upper())
        ("IR007_0.mat",     "IR"),       # Inner Race
        ("IR014_3.mat",     "IR"),
        ("OR014@6_1.mat",   "OR"),       # Outer Race (caractère spécial dans le nom)
        ("OR0216_2.mat",    "OR"),
        ("B021_2.mat",      "Ball"),     # Ball fault — B suivi d'un chiffre
        ("B007_0.mat",      "Ball"),
        ("xyz.mat",         "unknown"),  # préfixe non reconnu
        ("test123.mat",     "unknown"),
    ])
    def test_labels(self, filename: str, expected: str):
        assert detect_label(filename) == expected

    def test_b_followed_by_letter_is_unknown(self):
        """'Bfoo.mat' ne commence pas par B+chiffre → 'unknown', pas 'Ball'."""
        assert detect_label("Bfoo.mat") == "unknown"


# ---------------------------------------------------------------------------
# Tests split_anomaly_detection
# ---------------------------------------------------------------------------


class TestSplitAnomalyDetection:
    def test_train_size(self, df_multi_label: pd.DataFrame):
        """80% des 100 normales → 80 lignes dans le train."""
        df_train, _ = split_anomaly_detection(df_multi_label, train_ratio=0.8, seed=42)
        assert df_train.shape[0] == 80

    def test_train_contains_only_normal(self, df_multi_label: pd.DataFrame):
        """CRITIQUE : le train ne doit contenir aucun défaut."""
        df_train, _ = split_anomaly_detection(df_multi_label, train_ratio=0.8, seed=42)
        assert (df_train["label"] == "normal").all()

    def test_test_size(self, df_multi_label: pd.DataFrame):
        """Test = 20 normales restantes + 50 IR + 30 OR + 20 Ball = 120."""
        _, df_test = split_anomaly_detection(df_multi_label, train_ratio=0.8, seed=42)
        assert df_test.shape[0] == 120

    def test_test_label_counts(self, df_multi_label: pd.DataFrame):
        """Chaque label est présent en bonne quantité dans le test."""
        _, df_test = split_anomaly_detection(df_multi_label, train_ratio=0.8, seed=42)
        counts = df_test["label"].value_counts()
        assert counts["normal"] == 20
        assert counts["IR"]     == 50
        assert counts["OR"]     == 30
        assert counts["Ball"]   == 20

    def test_no_row_lost(self, df_multi_label: pd.DataFrame):
        """Train + test doit couvrir toutes les lignes originales."""
        df_train, df_test = split_anomaly_detection(df_multi_label, train_ratio=0.8, seed=42)
        assert len(df_train) + len(df_test) == len(df_multi_label)

    def test_reproducibility(self, df_multi_label: pd.DataFrame):
        """Deux appels avec le même seed produisent des splits identiques."""
        df_train_1, df_test_1 = split_anomaly_detection(df_multi_label, seed=42)
        df_train_2, df_test_2 = split_anomaly_detection(df_multi_label, seed=42)
        pd.testing.assert_frame_equal(df_train_1, df_train_2)
        pd.testing.assert_frame_equal(df_test_1,  df_test_2)

    def test_different_seeds_differ(self, df_multi_label: pd.DataFrame):
        """Deux seeds différents produisent des splits différents."""
        df_train_1, _ = split_anomaly_detection(df_multi_label, seed=0)
        df_train_2, _ = split_anomaly_detection(df_multi_label, seed=99)
        # Au moins un index diffère après le mélange aléatoire
        assert not df_train_1["source_file"].equals(df_train_2["source_file"]) or \
               not (df_train_1[[f"window_{i}" for i in range(N_WINDOW_COLS)]].values ==
                    df_train_2[[f"window_{i}" for i in range(N_WINDOW_COLS)]].values).all()


# ---------------------------------------------------------------------------
# Tests normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_range_fit(self, df_normal_only: pd.DataFrame):
        """Après fit, toutes les valeurs window_* doivent être dans [0, 1]."""
        df_norm, scaler = normalize(df_normal_only)
        cols = [c for c in df_norm.columns if c.startswith("window_")]
        vals = df_norm[cols].values
        assert vals.min() >= 0.0 - 1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_scaler_returned(self, df_normal_only: pd.DataFrame):
        """La fonction retourne bien un MinMaxScaler fitté."""
        _, scaler = normalize(df_normal_only)
        assert isinstance(scaler, MinMaxScaler)
        # Un scaler fitté possède l'attribut data_min_
        assert hasattr(scaler, "data_min_")

    def test_meta_columns_unchanged(self, df_normal_only: pd.DataFrame):
        """Les colonnes label et source_file restent intactes après normalisation."""
        df_norm, _ = normalize(df_normal_only)
        pd.testing.assert_series_equal(
            df_normal_only["label"].reset_index(drop=True),
            df_norm["label"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            df_normal_only["source_file"].reset_index(drop=True),
            df_norm["source_file"].reset_index(drop=True),
        )

    def test_no_leakage_scaler_fitted_on_train_only(self):
        """Le scaler ne doit être fitté que sur le train, jamais sur le test.

        Stratégie : train dans [-100, 0], test dans [50, 60].
        data_min_ doit correspondre aux min du train (≈ -100), pas du test (≈ 50).
        """
        rng = np.random.default_rng(1)
        cols = _make_window_cols(N_WINDOW_COLS)

        train_vals = rng.uniform(-100, 0, (60, N_WINDOW_COLS)).astype(np.float64)
        test_vals  = rng.uniform(50,  60, (40, N_WINDOW_COLS)).astype(np.float64)

        df_train = pd.DataFrame(train_vals, columns=cols)
        df_train["label"] = "normal"
        df_train["source_file"] = "Normal_0.mat"

        df_test = pd.DataFrame(test_vals, columns=cols)
        df_test["label"] = "IR"
        df_test["source_file"] = "IR007_0.mat"

        # Fit sur train uniquement
        _, scaler = normalize(df_train)

        # Vérification : data_min_ doit refléter les min du train
        train_mins = np.min(train_vals, axis=0)
        np.testing.assert_array_almost_equal(scaler.data_min_, train_mins, decimal=5)

        # Et PAS les min du test (qui seraient autour de 50)
        test_mins = np.min(test_vals, axis=0)
        assert not np.allclose(scaler.data_min_, test_mins), (
            "LEAKAGE détecté : le scaler semble fitté sur les données de test !"
        )

    def test_transform_uses_train_scaler(self):
        """Appliquer le scaler du train sur le test ne refit pas les statistiques."""
        rng = np.random.default_rng(2)
        cols = _make_window_cols(N_WINDOW_COLS)

        df_train = pd.DataFrame(rng.random((50, N_WINDOW_COLS)), columns=cols)
        df_train["label"] = "normal"
        df_train["source_file"] = "Normal_0.mat"

        df_test = pd.DataFrame(rng.random((30, N_WINDOW_COLS)) * 2, columns=cols)
        df_test["label"] = "IR"
        df_test["source_file"] = "IR007_0.mat"

        _, scaler_train = normalize(df_train)
        data_min_before = scaler_train.data_min_.copy()

        # Transform test avec le scaler du train
        _, scaler_after = normalize(df_test, scaler=scaler_train)

        # Le scaler retourné est le même objet (pas re-fitté)
        assert scaler_after is scaler_train
        # Ses statistiques n'ont pas changé
        np.testing.assert_array_equal(scaler_train.data_min_, data_min_before)
