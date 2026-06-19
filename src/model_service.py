import json
import os
import torch
from src.data_loader import VibrationPreprocessor
from src.models.autoencoder import get_model
from src.models.classifier import get_classifier

MODEL_PATH = "data/processed/best_model.pt"
SCALER_PATH = "data/processed/scaler.pkl"
THRESHOLD_PATH = "data/processed/threshold.json"
CLASSIFIER_PATH = "data/processed/best_classifier.pt"
CLASSIFIER_CFG_PATH = "data/processed/classifier_config.json"

_model = _preprocessor = _threshold_config = None
_classifier_model = _classifier_config = None


def load_all():
    global _model, _preprocessor, _threshold_config
    global _classifier_model, _classifier_config

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]):
        return False
    try:
        preprocessor = VibrationPreprocessor()
        preprocessor.load_scaler(SCALER_PATH)
        with open(THRESHOLD_PATH) as f:
            threshold_config = json.load(f)
        model = get_model(threshold_config["model_type"], height=128, width=32)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        _model, _preprocessor, _threshold_config = model, preprocessor, threshold_config

        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(CLASSIFIER_CFG_PATH):
            with open(CLASSIFIER_CFG_PATH) as f:
                _classifier_config = json.load(f)
            num_classes = _classifier_config.get("num_classes", 10)
            _classifier_model = get_classifier(num_classes)
            _classifier_model.load_state_dict(
                torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=True)
            )
            _classifier_model.eval()
            mtype = _classifier_config.get("model_type", "")
            print(f"[model_service] Classifier loaded: {mtype}")

        return True
    except Exception as e:
        print(f"[model_service] Erreur chargement : {e}")
        return False


def get_resources():
    if _model is None:
        load_all()
    return _model, _preprocessor, _threshold_config


def get_classifier_resources():
    if _classifier_model is None and _model is None:
        load_all()
    return _classifier_model, _classifier_config
