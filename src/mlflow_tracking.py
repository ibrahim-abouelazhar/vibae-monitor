"""
VibAE-Monitor — MLflow Tracking
=================================
Utilitaires pour logger les expériences d'entraînement dans MLflow.

Usage dans src/train.py (P3) :
    from src.mlflow_tracking import start_training_run, log_training_results

Usage dans l'API :
    from src.mlflow_tracking import log_inference
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np

# ── Configuration MLflow ───────────────────────────────────────────────────
MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_TRAIN   = "vibae-autoencoder-training"
EXPERIMENT_INFER   = "vibae-monitor-inference"

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


def setup_mlflow() -> None:
    """Initialise MLflow — à appeler une fois au démarrage."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"[MLflow] Tracking URI : {MLFLOW_URI}")


@contextmanager
def start_training_run(run_name: str = "cnn_autoencoder"):
    """Context manager pour un run d'entraînement.
    
    Usage :
        with start_training_run("cnn_1d_v2") as run:
            mlflow.log_param("epochs", 50)
            # ... entraînement ...
            mlflow.log_metric("roc_auc", 0.99)
    """
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TRAIN)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[MLflow] Run démarré : {run.info.run_id}")
        yield run
        print(f"[MLflow] Run terminé : {run.info.run_id}")


def log_training_results(
    run,
    model,
    history,
    threshold: float,
    metrics: dict[str, float],
    params: dict | None = None,
) -> None:
    """
    Log tous les résultats d'entraînement dans MLflow.
    
    À appeler depuis src/train.py (P3) après l'entraînement.

    Args:
        run       : objet run MLflow actif
        model     : modèle Keras entraîné
        history   : objet history.history de Keras
        threshold : seuil µ+3σ calculé
        metrics   : dict avec f1, roc_auc, recall, precision
        params    : hyperparamètres à logger (epochs, batch_size, etc.)
    """
    # ── Paramètres ──
    if params:
        mlflow.log_params(params)

    # ── Courbe de loss epoch par epoch ──
    for epoch, (loss, val_loss) in enumerate(
        zip(history["loss"], history.get("val_loss", []))
    ):
        mlflow.log_metrics(
            {"train_loss": loss, "val_loss": val_loss},
            step=epoch
        )

    # ── Métriques finales ──
    mlflow.log_metrics({
        "threshold":  threshold,
        "f1_score":   metrics.get("f1", 0.0),
        "roc_auc":    metrics.get("roc_auc", 0.0),
        "recall":     metrics.get("recall", 0.0),
        "precision":  metrics.get("precision", 0.0),
        "final_train_loss": history["loss"][-1],
        "final_val_loss":   history.get("val_loss", [0])[-1],
    })

    # ── Modèle ──
    mlflow.keras.log_model(model, artifact_path="autoencoder")
    print(f"[MLflow] Modèle loggé → run/{run.info.run_id}/autoencoder")

    # ── Sauvegarder les métriques en JSON (pour DVC) ──
    train_metrics = {
        "threshold":        threshold,
        "final_train_loss": history["loss"][-1],
        "final_val_loss":   history.get("val_loss", [0])[-1],
    }
    eval_metrics = {k: round(v, 4) for k, v in metrics.items()}

    (METRICS_DIR / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2))
    (METRICS_DIR / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    print("[MLflow] Métriques sauvegardées dans metrics/")


def log_inference(mse: float, status: str, latency_ms: float) -> None:
    """Log une inférence individuelle dans MLflow.
    
    Appelé en arrière-plan depuis api/main.py.
    Ne lève pas d'exception si MLflow est indisponible.
    """
    try:
        setup_mlflow()
        mlflow.set_experiment(EXPERIMENT_INFER)
        with mlflow.start_run(run_name="inference"):
            mlflow.log_metrics({
                "mse":        mse,
                "latency_ms": latency_ms,
                "is_anomaly": 1.0 if status == "ANOMALIE" else 0.0,
            })
    except Exception as exc:
        print(f"[MLflow] Warning: impossible de logger l'inférence ({exc})")


def get_best_run(metric: str = "roc_auc") -> dict | None:
    """Retourne le meilleur run MLflow selon une métrique donnée.
    
    Utile pour le réentraînement automatique : compare le nouveau modèle
    au meilleur existant avant de le déployer.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_TRAIN)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        best = runs[0]
        return {
            "run_id":  best.info.run_id,
            "metrics": best.data.metrics,
            "params":  best.data.params,
        }
    except Exception:
        return None


if __name__ == "__main__":
    # Test de connexion MLflow
    setup_mlflow()
    print("[MLflow] Test de connexion OK")
    best = get_best_run()
    if best:
        print(f"[MLflow] Meilleur run : {best['run_id']} — ROC-AUC={best['metrics'].get('roc_auc', 'N/A')}")
    else:
        print("[MLflow] Aucun run enregistré pour l'instant.")
