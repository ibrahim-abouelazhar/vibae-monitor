import os
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.models.classifier import get_classifier
from src import model_service  # FIXED P3
from src.data_loader import VibrationPreprocessor
from src.inference import run_inference  # FIXED P4
from backend.routers.live import router as live_router
from backend.routers.chunk import router as chunk_router
from backend.routers.stream import router as stream_router
from backend.routers.kafka_ws import router as kafka_ws_router

app = FastAPI(title="VibAE-Monitor API 2D", description="API pour la détection d'anomalies vibratoires via Autoencodeur Spectrogramme 2D")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resource paths
MODEL_PATH = model_service.MODEL_PATH  # FIXED P3
SCALER_PATH = model_service.SCALER_PATH  # FIXED P3
THRESHOLD_PATH = model_service.THRESHOLD_PATH  # FIXED P3
CLASSIFIER_PATH = "data/processed/best_classifier.pt"
CLASSIFIER_CONFIG_PATH = "data/processed/classifier_config.json"
BASELINE_PATH = "data/processed/baseline_model.pkl"
RAW_DATA_DIR = "data/raw"

model = None
preprocessor = None
threshold_config = None
classifier_model = None
classifier_config = None
baseline_model = None

class AnalysisRequest(BaseModel):
    window: list
    threshold_override: Optional[float] = None

class AnalysisResponse(BaseModel):
    mse: float
    status: str
    threshold: float
    reconstructed: list
    features: dict
    spectrogram: list
    recon_spectrogram: list
    fault_class: str = "Inconnue"
    baseline_class: str = "Inconnue"

def load_resources():
    global model, preprocessor, threshold_config, classifier_model, classifier_config, baseline_model
    if not model_service.load_resources():  # FIXED P3
        return False
    
    try:
        model, preprocessor, threshold_config = model_service.get_resources()  # FIXED P3
        
        # Load Classifier (if trained)
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(CLASSIFIER_CONFIG_PATH):
            with open(CLASSIFIER_CONFIG_PATH, "r") as f:
                classifier_config = json.load(f)
            classifier_model = get_classifier(classifier_config["num_classes"])
            classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=torch.device('cpu')))
            classifier_model.eval()
            print("Classifier loaded successfully.")
        else:
            classifier_model = None
            print("WARNING: Classifier weights not found. Multi-class fault diagnostics will be disabled.")
            
        # Load Baseline RF model (if trained)
        if os.path.exists(BASELINE_PATH):
            with open(BASELINE_PATH, "rb") as f:
                import pickle  # FIXED P3
                baseline_model = pickle.load(f)
            print("Baseline model loaded successfully.")
        else:
            baseline_model = None
            print("WARNING: Baseline RF model not found.")
            
        return True
    except Exception as e:
        print(f"Error loading resources: {e}")
        return False

@app.on_event("startup")
def startup_event():
    success = load_resources()
    if success:
        app.state.model = model
        app.state.preprocessor = preprocessor
        app.state.threshold_config = threshold_config
        app.state.classifier_model = classifier_model
        app.state.classifier_config = classifier_config
        app.state.baseline_model = baseline_model
        app.state.live_counters = {}
        app.state.loaded_windows = {}
        print("2D Model resources loaded successfully.")
    else:
        print("WARNING: Model and/or scaler files not found. Please train the 2D model first.")

@app.get("/health")
def health():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "waiting_for_model",
        "model_loaded": model_loaded,
        "classifier_loaded": classifier_model is not None,
        "baseline_loaded": baseline_model is not None,
        "threshold": threshold_config["threshold"] if threshold_config else None,
        "model_type": threshold_config["model_type"] if threshold_config else None,
        "window_size": threshold_config["window_size"] if threshold_config else None
    }

@app.get("/files")
def list_files():
    if not os.path.exists(RAW_DATA_DIR):
        raise HTTPException(status_code=404, detail=f"Directory '{RAW_DATA_DIR}' not found.")
        
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".mat")]
    
    file_list = []
    for f in files:
        label = f
        category = "Unknown"
        
        if "Normal" in f:
            category = "NORMAL"
            label = "État Normal (Moteur Sain)"
        elif "IR007" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Intérieure (0.007\")"
        elif "IR014" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Intérieure (0.014\")"
        elif "IR021" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Intérieure (0.021\")"
        elif "B007" in f:
            category = "DÉFAUT"
            label = "Défaut de Bille (0.007\")"
        elif "B014" in f:
            category = "DÉFAUT"
            label = "Défaut de Bille (0.014\")"
        elif "B021" in f:
            category = "DÉFAUT"
            label = "Défaut de Bille (0.021\")"
        elif "OR007" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Extérieure (0.007\")"
        elif "OR014" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Extérieure (0.014\")"
        elif "OR021" in f:
            category = "DÉFAUT"
            label = "Défaut Bague Extérieure (0.021\")"
            
        file_list.append({
            "filename": f,
            "label": label,
            "category": category
        })
        
    return file_list

@app.get("/files/{filename}/windows")
def get_file_windows(filename: str, limit: int = 300):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
        
    proc = preprocessor if preprocessor is not None else VibrationPreprocessor(window_size=2048, fs=48000)  # FIXED P1 P2
    
    try:
        signal_data = proc.load_de_signal(filepath)
        windows = proc.segment_signal(signal_data)
        num_windows = min(len(windows), limit)
        selected_windows = windows[:num_windows].tolist()
        return {
            "filename": filename,
            "total_windows": len(windows),
            "returned_windows": num_windows,
            "windows": selected_windows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file windows: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_window(request: AnalysisRequest):
    global model, preprocessor, threshold_config, classifier_model, baseline_model
    
    if model is None or preprocessor is None or threshold_config is None:
        success = load_resources()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Model or scaler files not loaded. Please run the training script first."
            )
            
    raw_window = request.window
    if len(raw_window) != threshold_config["window_size"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Le signal doit contenir exactement {threshold_config['window_size']} points (reçu: {len(raw_window)})."
        )
        
    raw_arr = np.array(raw_window)
    print(f"[DEBUG /analyze] len={len(raw_arr)}, min={raw_arr.min():.6f}, max={raw_arr.max():.6f}, mean={raw_arr.mean():.6f}, std={raw_arr.std():.6f}", flush=True)
    
    result = run_inference(preprocessor, model, threshold_config, raw_arr)  # FIXED P4
    magnitude_128 = result["spectrogram"]  # FIXED P4
    features = result["features"]  # FIXED P4
    mse = result["mse"]  # FIXED P4
    
    # Anomaly threshold check
    active_threshold = request.threshold_override if request.threshold_override is not None else threshold_config["threshold"]
    status = "ANOMALIE" if mse > active_threshold else "NORMAL"
    
    recon_magnitude_128 = result["recon_spectrogram"]  # FIXED P4
    reconstructed_1d = result["signal_reconstructed"]  # FIXED P4
    
    # 8. CNN Classifier Inference (Multi-class Diagnostics)
    fault_class = "Inconnue"
    if classifier_model is not None:
        tensor_32 = result["tensor_x"][:, :, :32, :]  # FIXED P6
        # Local min-max scaling
        t_min = tensor_32.min()
        t_max = tensor_32.max()
        tensor_32 = (tensor_32 - t_min) / (t_max - t_min + 1e-8)
        
        with torch.no_grad():
            outputs_clf = classifier_model(tensor_32)
            _, predicted_idx = outputs_clf.max(1)
            pred_idx = predicted_idx.item()
            
            # Map index to class label name
            IDX_TO_LABEL = {
                0: 'Normal',
                1: 'Ball_007', 2: 'Ball_014', 3: 'Ball_021',
                4: 'IR_007', 5: 'IR_014', 6: 'IR_021',
                7: 'OR_007', 8: 'OR_014', 9: 'OR_021'
            }
            fault_class = IDX_TO_LABEL.get(pred_idx, "Inconnue")
            
    # 9. Baseline Classifier Inference
    baseline_class = "Inconnue"
    if baseline_model is not None:
        features_array = np.array([[
            features["max"],
            features["min"],
            features["mean"],
            features["sd"],
            features["rms"],
            features["skewness"],
            features["kurtosis"],
            features["crest"],
            features["form"]
        ]])
        pred_label = baseline_model.predict(features_array)[0]
        baseline_class = pred_label
        
    return AnalysisResponse(
        mse=mse,
        status=status,
        threshold=active_threshold,
        reconstructed=reconstructed_1d.tolist(),
        features=features,
        spectrogram=magnitude_128.tolist(),
        recon_spectrogram=recon_magnitude_128.tolist(),
        fault_class=fault_class,
        baseline_class=baseline_class
    )

# Include the routers
app.include_router(live_router)
app.include_router(chunk_router, prefix="/predict")
app.include_router(stream_router)
app.include_router(kafka_ws_router)
