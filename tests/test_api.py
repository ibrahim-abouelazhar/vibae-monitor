"""
Tests de l'API FastAPI VibAE-Monitor.
Lance avec : python -m pytest tests/test_api.py -v
"""
import json

import joblib
import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from api.main import app, state

# ── Chargement des artefacts avant les tests ──────────────────────────────
state["scaler"]    = joblib.load("models/scaler.pkl")
state["model"]     = tf.keras.models.load_model("models/autoencoder.keras")
with open("models/threshold.json") as f:
    state["threshold"] = json.load(f)["threshold"]

client = TestClient(app)


def test_health():
    """GET /health doit retourner status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "scaler_loaded" in data
    assert "threshold" in data


def test_predict_normal_signal():
    """Un signal de faible amplitude doit retourner NORMAL."""
    signal = (np.random.normal(0, 0.01, 1024)).tolist()
    response = client.post("/predict", json={"signal": signal})
    assert response.status_code == 200
    data = response.json()
    assert "mse" in data
    assert "threshold" in data
    assert "status" in data
    assert data["status"] in ["NORMAL", "ANOMALIE"]
    assert data["confidence"] > 0


def test_predict_wrong_size():
    """Un signal de taille incorrecte doit retourner 422."""
    signal = [0.01] * 512   # trop court
    response = client.post("/predict", json={"signal": signal})
    assert response.status_code == 422


def test_predict_exact_size():
    """Un signal de 1024 points doit être accepté."""
    signal = [0.0] * 1024
    response = client.post("/predict", json={"signal": signal})
    assert response.status_code == 200


def test_predict_response_structure():
    """La réponse doit contenir tous les champs attendus."""
    signal = np.random.uniform(-1, 1, 1024).tolist()
    response = client.post("/predict", json={"signal": signal})
    assert response.status_code == 200
    data = response.json()
    required_fields = ["mse", "threshold", "status", "confidence"]
    for field in required_fields:
        assert field in data, f"Champ manquant : {field}"