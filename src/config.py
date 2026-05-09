"""Configuration centrale du projet vibae-monitor."""
from pathlib import Path

# === Chemins ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# === Paramètres signal ===
SAMPLING_RATE = 12000   # Hz - fichiers CWRU 12 kHz Drive End
WINDOW_SIZE = 1024      # points par fenêtre
OVERLAP = 0.5           # 50% de recouvrement

# === Split anomaly detection ===
TRAIN_RATIO = 0.8       # 80% des fenêtres normales pour train
RANDOM_SEED = 42

# === Mode développement ===
DEV_MODE = False        # True = ne charge qu'un sous-ensemble
DEV_SUBSET_SIZE = 4     # nombre de fichiers en DEV_MODE
