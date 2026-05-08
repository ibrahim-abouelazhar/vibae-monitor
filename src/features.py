import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
from scipy.signal import hilbert

# Importation des constantes depuis le fichier de configuration
# Ajustez ces imports selon la structure exacte de votre src/config.py
try:
    from src.config import SAMPLING_RATE, DATA_PROCESSED, PROJECT_ROOT
    # Définition du dossier de sortie pour les features s'il n'est pas dans config
    DATA_FEATURES = PROJECT_ROOT / "data" / "features"
except ImportError:
    # Valeurs par défaut de repli si config.py est inaccessible
    SAMPLING_RATE = 12000
    DATA_PROCESSED = Path("data/processed")
    DATA_FEATURES = Path("data/features")

def extract_frequency_features(df: pd.DataFrame, fs: int = SAMPLING_RATE) -> pd.DataFrame:
    """
    Extrait les caractéristiques dans le domaine fréquentiel pour chaque fenêtre d'un signal.
    Utilise la vectorisation NumPy/SciPy pour des performances optimales.
    
    Args:
        df: DataFrame contenant les colonnes 'window_0' à 'window_1023', 'label', 'source_file'.
        fs: Fréquence d'échantillonnage en Hz.
        
    Returns:
        pd.DataFrame: Nouveau DataFrame avec les caractéristiques calculées.
    """
    # 1. Isoler les données numériques des fenêtres
    window_cols = [c for c in df.columns if c.startswith('window_')]
    windows_data = df[window_cols].values
    N = windows_data.shape[1] # Devrait être 1024

    # 2. Calcul de la FFT et de la Densité Spectrale de Puissance (PSD)
    # rfft calcule la FFT pour des entrées réelles (renvoie seulement les fréquences positives)
    fft_vals = rfft(windows_data, axis=1)
    freqs = rfftfreq(N, 1/fs)
    psd = np.abs(fft_vals)**2 / N

    # Somme de la PSD pour la normalisation (éviter la division par zéro)
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_sum[psd_sum == 0] = 1e-10

    # 3. Extraction des caractéristiques spectrales
    
    # - Fréquence dominante (Hz)
    dominant_freq_idx = np.argmax(psd, axis=1)
    dominant_freq = freqs[dominant_freq_idx]

    # - Centroïde spectral (Hz)
    spectral_centroid = np.sum(freqs * psd, axis=1) / psd_sum.flatten()

    # - Entropie spectrale
    psd_norm = psd / psd_sum
    spectral_entropy = entropy(psd_norm, axis=1)

    # 4. Analyse d'enveloppe (Transformée de Hilbert)
    # Très utile pour détecter les modulations causées par les défauts (billes/bagues)
    analytic_signal = hilbert(windows_data, axis=1)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Retirer la composante continue (DC offset) de l'enveloppe avant la FFT
    envelope_no_dc = amplitude_envelope - np.mean(amplitude_envelope, axis=1, keepdims=True)
    env_fft = rfft(envelope_no_dc, axis=1)
    env_psd = np.abs(env_fft)**2 / N
    
    # Trouver la fréquence dominante de l'enveloppe
    env_dominant_freq_idx = np.argmax(env_psd, axis=1)
    env_dominant_freq = freqs[env_dominant_freq_idx]

    # 5. Construction du DataFrame de sortie
    features_df = pd.DataFrame({
        'dominant_freq_hz': dominant_freq,
        'spectral_centroid_hz': spectral_centroid,
        'spectral_entropy': spectral_entropy,
        'envelope_dominant_freq_hz': env_dominant_freq,
        'label': df['label'].values,
        'source_file': df['source_file'].values
    })

    return features_df

def process_and_save_features():
    """
    Charge les données traitées, extrait les caractéristiques et les sauvegarde.
    """
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    
    # Chemins des fichiers traités
    train_path = DATA_PROCESSED / "df_train.parquet"
    test_path = DATA_PROCESSED / "df_test.parquet"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Les fichiers traités sont introuvables. Lancez d'abord data_loader.py.")

    print("Chargement des données traitées (train et test)...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    print(f"Extraction des features pour l'ensemble Train ({len(df_train)} fenêtres)...")
    features_train = extract_frequency_features(df_train)
    
    print(f"Extraction des features pour l'ensemble Test ({len(df_test)} fenêtres)...")
    features_test = extract_frequency_features(df_test)
    
    # Sauvegarde
    out_train = DATA_FEATURES / "df_features_train.parquet"
    out_test = DATA_FEATURES / "df_features_test.parquet"
    
    features_train.to_parquet(out_train, index=False)
    features_test.to_parquet(out_test, index=False)
    
    print("✅ Extraction terminée avec succès !")
    print(f" - Train features sauvegardées : {out_train}")
    print(f" - Test features sauvegardées : {out_test}")

if __name__ == "__main__":
    process_and_save_features()