import json  # REFACTOR
import os  # REFACTOR
import torch  # REFACTOR
from src.data_loader import VibrationPreprocessor  # REFACTOR
from src.models.autoencoder import get_model  # REFACTOR
from src.models.classifier import get_classifier  # REFACTOR

MODEL_PATH = "data/processed/best_model.pt"  # REFACTOR
SCALER_PATH = "data/processed/scaler.pkl"  # REFACTOR
THRESHOLD_PATH = "data/processed/threshold.json"  # REFACTOR
CLASSIFIER_PATH = "data/processed/best_classifier.pt"  # REFACTOR
CLASSIFIER_CFG_PATH = "data/processed/classifier_config.json"  # REFACTOR

_model = _preprocessor = _threshold_config = None  # REFACTOR
_classifier_model = _classifier_config = None  # REFACTOR


def load_all():  # REFACTOR
    global _model, _preprocessor, _threshold_config  # REFACTOR
    global _classifier_model, _classifier_config  # REFACTOR

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]):  # REFACTOR
        return False  # REFACTOR
    try:  # REFACTOR
        preprocessor = VibrationPreprocessor()  # REFACTOR
        preprocessor.load_scaler(SCALER_PATH)  # REFACTOR
        with open(THRESHOLD_PATH) as f:  # REFACTOR
            threshold_config = json.load(f)  # REFACTOR
        model = get_model(threshold_config["model_type"], height=128, width=32)  # REFACTOR
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # REFACTOR
        model.eval()  # REFACTOR
        _model, _preprocessor, _threshold_config = model, preprocessor, threshold_config  # REFACTOR

        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(CLASSIFIER_CFG_PATH):  # REFACTOR
            with open(CLASSIFIER_CFG_PATH) as f:  # REFACTOR
                _classifier_config = json.load(f)  # REFACTOR
            num_classes = _classifier_config.get("num_classes", 10)  # REFACTOR
            _classifier_model = get_classifier(num_classes)  # REFACTOR — DenseNet or legacy CNN
            _classifier_model.load_state_dict(  # REFACTOR
                torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=True)  # REFACTOR
            )  # REFACTOR
            _classifier_model.eval()  # REFACTOR
            mtype = _classifier_config.get("model_type", "")  # REFACTOR
            print(f"[model_service] Classifier loaded: {mtype}")  # REFACTOR

        return True  # REFACTOR
    except Exception as e:  # REFACTOR
        print(f"[model_service] Erreur chargement : {e}")  # REFACTOR
        return False  # REFACTOR


def get_resources():  # REFACTOR
    if _model is None:  # REFACTOR
        load_all()  # REFACTOR
    return _model, _preprocessor, _threshold_config  # REFACTOR


def get_classifier_resources():  # REFACTOR
    if _classifier_model is None and _model is None:  # REFACTOR
        load_all()  # REFACTOR
    return _classifier_model, _classifier_config  # REFACTOR
