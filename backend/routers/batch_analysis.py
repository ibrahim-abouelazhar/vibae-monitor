import os
import numpy as np
import scipy.io as sio
from fastapi import APIRouter, HTTPException, Query
from src import model_service
from src.inference import run_inference
from src.severity import compute_severity

router = APIRouter()

RAW_DIR = "_archive/data/raw"

FAULT_TO_FILE = {
    "normal":   "Time_Normal_1_098.mat",
    "ball_007": "B007_1_123.mat",   "ball_014": "B014_1_190.mat",  "ball_021": "B021_1_227.mat",
    "ir_007":   "IR007_1_110.mat",  "ir_014":   "IR014_1_175.mat", "ir_021":   "IR021_1_214.mat",
    "or_007":   "OR007_6_1_136.mat","or_014":   "OR014_6_1_202.mat","or_021":  "OR021_6_1_239.mat",
}

TRUE_LABEL = {
    "normal":   "Normal",
    "ball_007": "Ball_007", "ball_014": "Ball_014", "ball_021": "Ball_021",
    "ir_007":   "IR_007",   "ir_014":   "IR_014",   "ir_021":   "IR_021",
    "or_007":   "OR_007",   "or_014":   "OR_014",   "or_021":   "OR_021",
}


def _load_signal(filename: str) -> np.ndarray:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mat = sio.loadmat(path)
    de_keys = [k for k in mat.keys() if "DE_time" in k]
    if not de_keys:
        raise ValueError(f"No DE_time in {filename}")
    return mat[de_keys[0]].flatten().astype(np.float32)


@router.get("/run")
def run_batch(
    fault_type: str = Query(..., description="Clé de défaut (ex: normal, ball_007, or_021)"),
    max_windows: int = Query(100, ge=10, le=500),
):
    """
    Analyse complète d'un signal CWRU par fenêtres successives.
    Retourne : profil MSE, taux d'anomalie, distribution des prédictions, précision.
    """
    if fault_type not in FAULT_TO_FILE:
        raise HTTPException(400, f"fault_type inconnu. Valeurs : {list(FAULT_TO_FILE.keys())}")

    model, preprocessor, threshold_config = model_service.get_resources()
    classifier_model, classifier_config = model_service.get_classifier_resources()
    if model is None:
        raise HTTPException(503, "Modèle non chargé.")

    try:
        sig = _load_signal(FAULT_TO_FILE[fault_type])
    except FileNotFoundError as e:
        raise HTTPException(404, f"Fichier signal introuvable : {e}")

    window_size = int(threshold_config["window_size"])
    step_size   = int(threshold_config.get("step_size", window_size))

    # Segment signal
    starts = list(range(0, len(sig) - window_size + 1, step_size))[:max_windows]
    if not starts:
        raise HTTPException(422, "Signal trop court pour extraire une fenêtre.")

    mse_values, sev_scores, predictions, is_anomaly_list = [], [], [], []
    true_label = TRUE_LABEL.get(fault_type, "Unknown")
    correct = 0

    for start in starts:
        window = sig[start:start + window_size]
        res = run_inference(preprocessor, model, threshold_config,
                            classifier_model, classifier_config, window)
        mse_values.append(res["mse"])
        sev_scores.append(res["severity_score"])
        is_anomaly_list.append(res["is_anomaly"])
        predictions.append(res["predicted_fault"])

        # Accuracy: does prediction match true label category?
        pred = res["predicted_fault"]
        if fault_type == "normal" and pred == "Normal":
            correct += 1
        elif fault_type != "normal" and pred != "Normal":
            correct += 1

    total = len(starts)
    anomaly_count = sum(is_anomaly_list)

    # Class distribution
    class_counts: dict[str, int] = {}
    for p in predictions:
        class_counts[p] = class_counts.get(p, 0) + 1

    # Majority prediction
    majority = max(class_counts, key=class_counts.get) if class_counts else "Unknown"

    return {
        "fault_type": fault_type,
        "true_label": true_label,
        "total_windows": total,
        "window_size": window_size,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / total, 4),
        "accuracy": round(correct / total, 4),
        "majority_prediction": majority,
        "mean_mse": round(float(np.mean(mse_values)), 6),
        "mean_severity": round(float(np.mean(sev_scores)), 1),
        "threshold": threshold_config["threshold"],
        "window_indices": list(range(total)),
        "mse_values": mse_values,
        "severity_values": sev_scores,
        "is_anomaly_list": is_anomaly_list,
        "class_counts": class_counts,
        "predictions": predictions,
    }
