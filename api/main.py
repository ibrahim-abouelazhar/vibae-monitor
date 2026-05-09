"""
VibAE-Monitor — API FastAPI
============================
Endpoint principal : POST /predict
  - Reçoit une fenêtre de signal (1024 points)
  - Normalise avec le scaler de P1 (models/scaler.pkl)
  - Passe dans l'autoencoder de P3 (models/autoencoder.keras)
  - Calcule le MSE de reconstruction
  - Compare au seuil µ+3σ → NORMAL / ANOMALIE

Lancement :
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docker :
    docker-compose up --build
"""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PredictRequest, PredictResponse, RetrainResponse

# ── Chemins (relatifs à la racine du projet) ──────────────────────────────
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    "models"))
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
MODEL_PATH    = MODELS_DIR / "autoencoder.keras"   # livré par P3
THRESHOLD_PATH = MODELS_DIR / "threshold.json"      # livré par P3

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "vibae-monitor-inference"

# ── État global de l'application ───────────────────────────────────────────
state: dict = {
    "model":     None,
    "scaler":    None,
    "threshold": None,
}


# ── Chargement au démarrage ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle, le scaler et le seuil au démarrage du serveur."""
    print("[VibAE] Chargement des artefacts…")

    # Scaler — obligatoire (livré par P1)
    if not SCALER_PATH.exists():
        raise RuntimeError(
            f"scaler.pkl introuvable : {SCALER_PATH}\n"
            "Lancez d'abord : python -m src.data_loader"
        )
    state["scaler"] = joblib.load(SCALER_PATH)
    print(f"[VibAE] ✅ Scaler chargé  ({SCALER_PATH})")

    # Modèle — placeholder si P3 n'a pas encore livré
    if MODEL_PATH.exists():
        import tensorflow as tf
        state["model"] = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"[VibAE] ✅ Modèle chargé ({MODEL_PATH})")
    else:
        print(f"[VibAE] ⚠️  Modèle absent ({MODEL_PATH}) — mode PLACEHOLDER actif")
        state["model"] = None

    # Seuil µ+3σ — placeholder si P3 n'a pas encore livré
    if THRESHOLD_PATH.exists():
        import json
        with open(THRESHOLD_PATH) as f:
            meta = json.load(f)
        state["threshold"] = meta["threshold"]
        print(f"[VibAE] OK Seuil charge : {state['threshold']:.6f}")
    else:
        state["threshold"] = 0.0003   # valeur du notebook de test
        print(f"[VibAE] ⚠️  threshold.pkl absent — valeur par défaut : {state['threshold']}")

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"[VibAE] ✅ MLflow  → {MLFLOW_URI}")

    yield   # ← serveur actif ici

    print("[VibAE] Arrêt du serveur.")


# ── Application ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="VibAE-Monitor API",
    description="Détection d'anomalies vibratoires par CNN 1D Autoencoder — CWRU Dataset",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # P5 (Streamlit) peut appeler depuis n'importe quelle origine
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normalise un signal (1024,) avec le scaler MinMax de P1."""
    scaler = state["scaler"]
    # Le scaler a été fitté sur (N, 1024) → reshape en (1, 1024)
    return scaler.transform(signal.reshape(1, -1))   # (1, 1024)


def _predict_mse(signal_norm: np.ndarray) -> tuple[float, np.ndarray]:
    """Passe le signal dans l'autoencoder et retourne (mse, signal_reconstruit)."""
    model = state["model"]

    if model is None:
        # ── PLACEHOLDER : MSE aléatoire réaliste jusqu'à livraison P3 ──
        mse = float(np.random.uniform(1e-5, 8e-4))
        reconstructed = signal_norm + np.random.normal(0, 0.01, signal_norm.shape)
        return mse, reconstructed

    # ── Vrai modèle Keras CNN 1D ──
    # Reshape (1, 1024) → (1, 1024, 1)  attendu par Conv1D
    x = signal_norm.reshape(1, 1024, 1)
    x_hat = model.predict(x, verbose=0)                     # (1, 1024, 1)
    mse = float(np.mean((x - x_hat) ** 2))
    return mse, x_hat.reshape(1, 1024)


def _log_inference(mse: float, status: str, latency_ms: float) -> None:
    """Log les métriques d'inférence dans MLflow (non bloquant)."""
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metrics({
                "mse":            mse,
                "latency_ms":     latency_ms,
                "is_anomaly":     1.0 if status == "ANOMALIE" else 0.0,
            })
    except Exception:
        pass   # Ne pas bloquer l'API si MLflow est indisponible


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie que l'API est opérationnelle et que les artefacts sont chargés."""
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
        scaler_loaded=state["scaler"] is not None,
        threshold=state["threshold"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inférence"])
def predict(body: PredictRequest, background_tasks: BackgroundTasks):
    """
    Détecte si une fenêtre de signal vibratoire est NORMAL ou ANOMALIE.

    **Corps de la requête** :
    ```json
    { "signal": [0.01, 0.02, -0.01, ...] }   // exactement 1024 floats
    ```

    **Réponse** :
    ```json
    {
      "mse": 0.000123,
      "threshold": 0.000294,
      "status": "NORMAL",
      "confidence": 0.42
    }
    ```
    """
    if state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Scaler non chargé")

    t0 = time.perf_counter()

    # 1. Conversion + normalisation
    signal = np.array(body.signal, dtype=np.float32)   # (1024,)
    signal_norm = _normalize(signal)                    # (1, 1024)

    # 2. Reconstruction + MSE
    mse, _ = _predict_mse(signal_norm)

    # 3. Décision
    threshold  = state["threshold"]
    status     = "ANOMALIE" if mse > threshold else "NORMAL"
    confidence = round(mse / threshold, 4)

    latency_ms = (time.perf_counter() - t0) * 1000

    # 4. Log MLflow en arrière-plan (ne bloque pas la réponse)
    background_tasks.add_task(_log_inference, mse, status, latency_ms)

    return PredictResponse(
        mse=round(mse, 8),
        threshold=round(threshold, 8),
        status=status,
        confidence=confidence,
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["MLOps"])
def retrain(background_tasks: BackgroundTasks):
    """
    Déclenche manuellement un cycle de réentraînement.
    Lance le pipeline DVC en arrière-plan et log les métriques dans MLflow.
    
    ⚠️ Nécessite que data/processed/ soit à jour (relancer data_loader.py d'abord).
    """
    import subprocess

    def _run_retrain():
        """Exécute le pipeline DVC + log MLflow."""
        with mlflow.start_run(run_name="retrain") as run:
            try:
                # Exécution du pipeline DVC
                result = subprocess.run(
                    ["dvc", "repro"],
                    capture_output=True, text=True, timeout=600
                )
                mlflow.log_param("dvc_status", "success" if result.returncode == 0 else "failed")
                mlflow.log_text(result.stdout, "dvc_output.txt")

                # Rechargement du modèle et du seuil mis à jour
                if MODEL_PATH.exists():
                    import tensorflow as tf
                    state["model"] = tf.keras.models.load_model(str(MODEL_PATH))

                if THRESHOLD_PATH.exists():
                    import json
                    with open(THRESHOLD_PATH) as f:
                        meta = json.load(f)
                    state["threshold"] = meta["threshold"]
                    mlflow.log_metric("threshold", state["threshold"])

                print(f"[VibAE] Réentraînement terminé — run_id={run.info.run_id}")

            except subprocess.TimeoutExpired:
                mlflow.log_param("dvc_status", "timeout")

    run_id = f"retrain_{int(time.time())}"
    background_tasks.add_task(_run_retrain)

    return RetrainResponse(
        status="Réentraînement lancé en arrière-plan",
        run_id=run_id,
        metrics={"threshold": state["threshold"] or 0.0},
    )


# ── Point d'entrée direct ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)