from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.inference import run_inference
from src import model_service
from src.database import save_inference
from src.severity import update_trend
from backend.routers.simulator import _machine_states as _sim_states

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChunkRequest(BaseModel):
    samples: List[float]
    machine_id: Optional[str] = "machine_1"


class BatchRequest(BaseModel):
    samples: List[float]


class TrendInfo(BaseModel):
    direction: str       # "rising" | "stable" | "falling"
    slope: float
    alert: bool


class InferenceResult(BaseModel):
    mse: float
    threshold: float
    is_anomaly: bool
    severity_score: float
    predicted_fault: str
    rms: float
    kurtosis: float
    variance: float
    fft_peak_hz: float
    fft_freqs: List[float]
    fft_amplitudes: List[float]
    spectrogram: List[List[float]]
    signal_reconstructed: List[float]
    trend: TrendInfo


class ChunkResponse(BaseModel):
    machine_id: str
    buffer_fill: int
    window_size: int
    inference_triggered: bool
    result: Optional[InferenceResult] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resources_or_503():
    model, preprocessor, threshold_config = model_service.get_resources()
    classifier_model, classifier_config = model_service.get_classifier_resources()
    if model is None or preprocessor is None or threshold_config is None:
        raise HTTPException(status_code=503,
                            detail="Model, scaler or threshold configuration not loaded.")
    return model, preprocessor, threshold_config, classifier_model, classifier_config


def _format_result(result: dict, machine_id: str) -> InferenceResult:
    features = result["features"]

    # Trend (in-memory rolling window)
    trend_data = update_trend(machine_id, result["mse"])

    return InferenceResult(
        mse=result["mse"],
        threshold=result["threshold"],
        is_anomaly=result["is_anomaly"],
        severity_score=result["severity_score"],
        predicted_fault=result["predicted_fault"],
        rms=features["rms"],
        kurtosis=features["kurtosis"],
        variance=features["variance"],
        fft_peak_hz=features["peak_frequency"],
        fft_freqs=features["fft_freqs"],
        fft_amplitudes=features["fft_amps"],
        spectrogram=result["spectrogram"].tolist(),
        signal_reconstructed=result["signal_reconstructed"].tolist(),
        trend=TrendInfo(
            direction=trend_data["direction"],
            slope=trend_data["slope"],
            alert=trend_data["alert"],
        ),
    )


def _persist(machine_id: str, result: dict):
    """Save inference to SQLite (non-blocking best-effort)."""
    try:
        fault_type = _sim_states.get(machine_id, {}).get("fault_type", "unknown")
        save_inference(
            machine_id=machine_id,
            fault_type=fault_type,
            mse=result["mse"],
            threshold=result["threshold"],
            is_anomaly=result["is_anomaly"],
            predicted_fault=result["predicted_fault"],
            severity_score=result["severity_score"],
            features=result["features"],
        )
    except Exception as e:
        print(f"[chunk] DB save warning: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chunk", response_model=ChunkResponse)
async def predict_chunk(payload: ChunkRequest, request: Request):
    model, preprocessor, threshold_config, classifier_model, classifier_config = _resources_or_503()
    machine_id = payload.machine_id or "machine_1"
    window_size = int(threshold_config["window_size"])

    if not hasattr(request.app.state, "buffers"):
        request.app.state.buffers = {}
    buffer = request.app.state.buffers.setdefault(machine_id, [])
    buffer.extend(float(s) for s in payload.samples)

    if len(buffer) < window_size:
        return ChunkResponse(
            machine_id=machine_id,
            buffer_fill=len(buffer),
            window_size=window_size,
            inference_triggered=False,
        )

    raw_window = np.asarray(buffer[:window_size], dtype=np.float32)
    request.app.state.buffers[machine_id] = []

    result = run_inference(preprocessor, model, threshold_config,
                           classifier_model, classifier_config, raw_window)
    _persist(machine_id, result)
    formatted = _format_result(result, machine_id)

    return ChunkResponse(
        machine_id=machine_id,
        buffer_fill=0,
        window_size=window_size,
        inference_triggered=True,
        result=formatted,
    )


@router.post("/batch", response_model=InferenceResult)
async def predict_batch(payload: BatchRequest):
    model, preprocessor, threshold_config, classifier_model, classifier_config = _resources_or_503()
    window_size = int(threshold_config["window_size"])
    if len(payload.samples) != window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Le signal doit contenir exactement {window_size} points.",
        )
    raw_window = np.asarray(payload.samples, dtype=np.float32)
    result = run_inference(preprocessor, model, threshold_config,
                           classifier_model, classifier_config, raw_window)
    return _format_result(result, "batch")
