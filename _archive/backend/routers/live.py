import os
import torch
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from src.model_service import get_resources  # FIXED P3

router = APIRouter()

RAW_DATA_DIR = "data/raw"

class LiveRequest(BaseModel):
    file: str
    window_index: Optional[int] = None

class LiveResponse(BaseModel):
    window_index: int
    total_windows: int
    signal_1d: list
    spectrogram: list
    signal_reconstructed: list
    fft_freqs: list
    fft_amplitudes: list
    fft_peak_hz: float
    mse: float
    threshold: float
    is_anomaly: bool
    rms: float
    kurtosis: float
    variance: float
    timestamp: str


def get_shared_resources(request: Request):
    """
    Helper to extract model, preprocessor, and threshold config from app state.
    """
    app = request.app
    model = getattr(app.state, "model", None)
    preprocessor = getattr(app.state, "preprocessor", None)
    threshold_config = getattr(app.state, "threshold_config", None)
    
    if model is None or preprocessor is None or threshold_config is None:
        model, preprocessor, threshold_config = get_resources()  # FIXED P3
        if model and preprocessor and threshold_config:
            app.state.model = model
            app.state.preprocessor = preprocessor
            app.state.threshold_config = threshold_config
            if not hasattr(app.state, "live_counters"):
                app.state.live_counters = {}
            if not hasattr(app.state, "loaded_windows"):
                app.state.loaded_windows = {}
                
    return model, preprocessor, threshold_config


@router.post("/predict/live", response_model=LiveResponse)
def predict_live(request: Request, payload: LiveRequest):
    model, preprocessor, threshold_config = get_shared_resources(request)
    
    if model is None or preprocessor is None or threshold_config is None:
        raise HTTPException(
            status_code=503,
            detail="Model resources not initialized on the server yet."
        )

    filename = payload.file
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found."
        )

    # Cache loaded windows per file in app state for performance
    if not hasattr(request.app.state, "loaded_windows"):
        request.app.state.loaded_windows = {}
    
    if filename not in request.app.state.loaded_windows:
        try:
            signal_data = preprocessor.load_de_signal(filepath)
            windows = preprocessor.segment_signal(signal_data)
            request.app.state.loaded_windows[filename] = windows
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading and segmenting file {filename}: {str(e)}"
            )

    windows = request.app.state.loaded_windows[filename]
    num_windows = len(windows)
    if num_windows == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No vibration windows could be segmented from file {filename}."
        )

    # Initialize live counters in app state if not present
    if not hasattr(request.app.state, "live_counters"):
        request.app.state.live_counters = {}
    
    # Check if window_index is explicitly requested
    if payload.window_index is not None:
        idx = payload.window_index
    else:
        # Retrieve the next index from server state
        idx = request.app.state.live_counters.get(filename, 0)

    # Wrap around if index goes out of bounds
    if idx < 0 or idx >= num_windows:
        idx = 0

    raw_window = windows[idx]
    
    # 1. Compute STFT (Magnitude & Phase)
    magnitude_128, original_stft = preprocessor.compute_stft(raw_window)
    
    # 2. Extract 1D features
    features = preprocessor.extract_features(raw_window)
    
    # 3. Preprocess / Scale Magnitude
    scaled_magnitude = preprocessor.transform_magnitude(magnitude_128)
    
    # 4. Prepare Tensor for 2D Conv PyTorch (batch, channel, height, width)
    tensor_x = torch.tensor(scaled_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 5. Inference
    with torch.no_grad():
        tensor_y = model(tensor_x)
        
    # 6. Calculate reconstruction MSE on spectrogram magnitude
    mse = torch.mean((tensor_x - tensor_y) ** 2).item()
    
    # Status prediction
    threshold = threshold_config["threshold"]
    is_anomaly = mse > threshold
    
    # 7. Squeeze and inverse scale reconstructed magnitude
    recon_scaled = tensor_y.squeeze(0).squeeze(0).cpu().numpy()
    recon_magnitude_128 = preprocessor.inverse_transform_magnitude(recon_scaled)
    
    # 8. Reconstruct 1D signal using original phase (iSTFT)
    reconstructed_1d = preprocessor.reconstruct_signal_from_stft(recon_magnitude_128, original_stft)
    
    # Increment counter for this file in app state
    request.app.state.live_counters[filename] = (idx + 1) % num_windows

    # Prepare timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return LiveResponse(
        window_index=idx,
        total_windows=num_windows,
        signal_1d=raw_window.tolist(),
        spectrogram=magnitude_128.tolist(),
        signal_reconstructed=reconstructed_1d.tolist(),
        fft_freqs=features["fft_freqs"],
        fft_amplitudes=features["fft_amps"],
        fft_peak_hz=features["peak_frequency"],
        mse=mse,
        threshold=threshold,
        is_anomaly=is_anomaly,
        rms=features["rms"],
        kurtosis=features["kurtosis"],
        variance=features["variance"],
        timestamp=now_str
    )
