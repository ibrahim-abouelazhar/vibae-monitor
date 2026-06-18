import os
import json
import asyncio
import datetime
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional

from src.inference import run_inference  # FIXED P4
from src.model_service import get_resources  # FIXED P3

router = APIRouter()

RAW_DATA_DIR = "data/raw"

class StreamState:
    def __init__(self):
        self.file: str = "Time_Normal_1_098.mat"
        self.signal_data: Optional[np.ndarray] = None
        self.cursor: int = 0
        self.task: Optional[asyncio.Task] = None
        self.queues: set[asyncio.Queue] = set()

stream_state = StreamState()

async def signal_producer():
    """
    Single global background task that paces the 1D physical vibration signal
    at the CWRU sampling rate (48000 Hz) using the active StreamState signal.  # FIXED P1
    """
    chunk_size = 64
    sample_rate = 48000  # FIXED P1
    sleep_time = 0.02  # 20ms per chunk — 4× slower than real-time (5ms) for readable UI scrolling
    
    while True:
        try:
            # Check if signal is loaded. If not, try loading the active file.
            if stream_state.signal_data is None:
                filepath = os.path.join(RAW_DATA_DIR, stream_state.file)
                if os.path.exists(filepath):
                    _, preprocessor, _ = get_resources()  # FIXED P3
                    stream_state.signal_data = preprocessor.load_de_signal(filepath)
                    stream_state.cursor = 0
                else:
                    await asyncio.sleep(0.1)
                    continue
            
            # Retrieve signal and cursor references locally to handle swaps cleanly
            signal = stream_state.signal_data
            cursor = stream_state.cursor
            
            # Wrap around and loop from start of signal if next chunk is out-of-bounds
            if cursor + chunk_size > len(signal):
                stream_state.cursor = 0
                continue
                
            chunk = signal[cursor : cursor + chunk_size].tolist()
            
            # Broadcast the chunk to all active client queues
            for q in list(stream_state.queues):
                try:
                    q.put_nowait({"type": "chunk", "data": chunk, "cursor": cursor, "file": stream_state.file})
                except asyncio.QueueFull:
                    pass
                    
            stream_state.cursor += chunk_size
            await asyncio.sleep(sleep_time)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in global producer loop: {e}")
            await asyncio.sleep(0.1)

@router.get("/predict/stream")
async def predict_stream(file: str, request: Request):
    app = request.app
    
    # Check resources are loaded
    model = getattr(app.state, "model", None)
    preprocessor = getattr(app.state, "preprocessor", None)
    threshold_config = getattr(app.state, "threshold_config", None)
    
    if model is None or preprocessor is None or threshold_config is None:
        model, preprocessor, threshold_config = get_resources()  # FIXED P3
        if model is None or preprocessor is None or threshold_config is None:  # FIXED P3
            raise HTTPException(status_code=503, detail="Model assets not loaded.")  # FIXED P3
        app.state.model = model  # FIXED P3
        app.state.preprocessor = preprocessor  # FIXED P3
        app.state.threshold_config = threshold_config  # FIXED P3

    # Swap file on the global state if the requested file is different
    if stream_state.file != file or stream_state.signal_data is None:
        filepath = os.path.join(RAW_DATA_DIR, file)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File {file} not found in {RAW_DATA_DIR}")
        try:
            stream_state.signal_data = preprocessor.load_de_signal(filepath)
            stream_state.file = file
            stream_state.cursor = 0
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading CWRU file: {e}")

    # Create queue for this client connection
    q = asyncio.Queue(maxsize=200)
    stream_state.queues.add(q)
    
    # Start global background producer task if not currently running
    if stream_state.task is None or stream_state.task.done():
        stream_state.task = asyncio.create_task(signal_producer())
        
    async def event_generator():
        # Buffer to accumulate samples for running inference
        history = []
        new_samples_count = 0
        
        try:
            while True:
                # Check for disconnection
                if await request.is_disconnected():
                    break
                    
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send keep-alive ping
                    yield ": ping\n\n"
                    continue
                    
                chunk = event["data"]
                cursor = event.get("cursor", 0)
                active_file = event.get("file", stream_state.file)
                
                # Push raw chunk to frontend
                yield f"event: chunk\ndata: {json.dumps(chunk)}\n\n"
                
                # Accumulate for windowed CNN inference
                history.extend(chunk)
                new_samples_count += len(chunk)
                
                # Keep history size bounded
                if len(history) > 3000:
                    history = history[-3000:]
                    
                if new_samples_count >= 1024:
                    new_samples_count -= 1024
                    
                    # Window size of 2048 is required by the preprocessor to perform STFT (128x32).  # FIXED P2
                    if len(history) >= 2048:  # FIXED P2
                        raw_window = history[-2048:]  # FIXED P2
                    else:
                        raw_window = history + [0.0] * (2048 - len(history))  # FIXED P2
                        
                    raw_arr = np.array(raw_window)
                    result = run_inference(preprocessor, model, threshold_config, raw_arr)  # FIXED P4
                    magnitude_128 = result["spectrogram"]  # FIXED P4
                    features = result["features"]  # FIXED P4
                    reconstructed_1d = result["signal_reconstructed"]  # FIXED P4
                    recon_1024 = reconstructed_1d[-1024:].tolist()

                    mse = result["mse"]  # FIXED P4
                    threshold = result["threshold"]  # FIXED P4
                    is_anomaly = result["is_anomaly"]  # FIXED P4
                    
                    # Determine total windows in file for metrics response
                    total_windows = getattr(app.state, f"len_{active_file}", 200)
                    if not hasattr(app.state, f"len_{active_file}"):
                        try:
                            filepath = os.path.join(RAW_DATA_DIR, active_file)
                            sig_data = preprocessor.load_de_signal(filepath)
                            total_windows = len(sig_data) // 1024
                            setattr(app.state, f"len_{active_file}", total_windows)
                        except Exception:
                            pass
                            
                    connection_window_index = (cursor // 1024) + 1
                    
                    metrics = {
                        "window_index": connection_window_index,
                        "total_windows": total_windows,
                        "spectrogram": magnitude_128.tolist(),
                        "fft_freqs": features["fft_freqs"],
                        "fft_amplitudes": features["fft_amps"],
                        "fft_peak_hz": features["peak_frequency"],
                        "mse": mse,
                        "threshold": threshold,
                        "is_anomaly": is_anomaly,
                        "rms": features["rms"],
                        "kurtosis": features["kurtosis"],
                        "variance": features["variance"],
                        "signal_reconstructed": recon_1024,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    }
                    
                    yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
                    
        finally:
            # Clean up queue when client connection closes
            stream_state.queues.discard(q)
            if len(stream_state.queues) == 0:
                if stream_state.task:
                    stream_state.task.cancel()
                    stream_state.task = None
                    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/predict/stream/switch")
async def predict_stream_switch(file: str):
    """
    Atomically swap the active source file in StreamState to allow continuous streaming.
    """
    filepath = os.path.join(RAW_DATA_DIR, file)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File {file} not found in {RAW_DATA_DIR}")
        
    try:
        _, preprocessor, _ = get_resources()  # FIXED P3
        new_signal = preprocessor.load_de_signal(filepath)
        
        # Swap atomically on the global state
        stream_state.signal_data = new_signal
        stream_state.file = file
        stream_state.cursor = 0
        
        return {"status": "switched", "file": file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching file: {e}")

@router.post("/predict/stream/stop")
async def predict_stream_stop(file: Optional[str] = None):
    """
    Stop the global background producer task and disconnect all client queues.
    """
    if stream_state.task:
        stream_state.task.cancel()
        try:
            await stream_state.task
        except asyncio.CancelledError:
            pass
        stream_state.task = None
        
    stream_state.queues.clear()
    return {"status": "stopped"}
