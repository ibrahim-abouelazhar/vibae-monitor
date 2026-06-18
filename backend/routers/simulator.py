import os
import numpy as np
import scipy.io as sio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

RAW_DIR = "_archive/data/raw"

FAULT_TO_FILE = {
    "normal":    "Time_Normal_1_098.mat",
    "ball_007":  "B007_1_123.mat",
    "ball_014":  "B014_1_190.mat",
    "ball_021":  "B021_1_227.mat",
    "ir_007":    "IR007_1_110.mat",
    "ir_014":    "IR014_1_175.mat",
    "ir_021":    "IR021_1_214.mat",
    "or_007":    "OR007_6_1_136.mat",
    "or_014":    "OR014_6_1_202.mat",
    "or_021":    "OR021_6_1_239.mat",
}

# In-memory signal cache: {filename: np.ndarray}
_signals: dict = {}
# Per-machine state: {machine_id: {"fault_type": str, "cursor": int}}
_machine_states: dict = {}

MACHINE_IDS = ["machine_1", "machine_2", "machine_3", "machine_4"]


def _load_signal(filename: str) -> np.ndarray:
    if filename in _signals:
        return _signals[filename]
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signal file not found: {path}")
    mat = sio.loadmat(path)
    de_keys = [k for k in mat.keys() if "DE_time" in k]
    if not de_keys:
        raise ValueError(f"No DE_time key in {filename}")
    sig = mat[de_keys[0]].flatten().astype(np.float32)
    _signals[filename] = sig
    return sig


def load_all_signals():
    """Pre-load all signals at startup."""
    for fault, fname in FAULT_TO_FILE.items():
        try:
            _load_signal(fname)
        except Exception as e:
            print(f"[simulator] Warning — could not load {fname}: {e}")
    for mid in MACHINE_IDS:
        _machine_states[mid] = {"fault_type": "normal", "cursor": 0}
    print(f"[simulator] {len(_signals)} signal file(s) loaded.")


class SimStateRequest(BaseModel):
    machine_id: str
    fault_type: str  # one of FAULT_TO_FILE keys


@router.post("/state")
def set_machine_state(req: SimStateRequest):
    if req.fault_type not in FAULT_TO_FILE:
        raise HTTPException(400, f"fault_type must be one of: {list(FAULT_TO_FILE.keys())}")
    if req.machine_id not in MACHINE_IDS:
        raise HTTPException(400, f"machine_id must be one of: {MACHINE_IDS}")
    _machine_states[req.machine_id] = {"fault_type": req.fault_type, "cursor": 0}
    return {"machine_id": req.machine_id, "fault_type": req.fault_type}


@router.get("/chunk")
def get_chunk(
    machine_id: str = Query(...),
    n: int = Query(256, ge=64, le=2048),
):
    if machine_id not in _machine_states:
        _machine_states[machine_id] = {"fault_type": "normal", "cursor": 0}

    state = _machine_states[machine_id]
    filename = FAULT_TO_FILE.get(state["fault_type"], FAULT_TO_FILE["normal"])

    try:
        sig = _load_signal(filename)
    except FileNotFoundError:
        raise HTTPException(503, "Signal data files not available.")

    cursor = state["cursor"]
    total = len(sig)
    end = cursor + n

    if end <= total:
        chunk = sig[cursor:end]
        state["cursor"] = end % total
    else:
        # Wrap around
        part1 = sig[cursor:]
        part2 = sig[: (end % total)]
        chunk = np.concatenate([part1, part2])
        state["cursor"] = end % total

    return {
        "machine_id": machine_id,
        "fault_type": state["fault_type"],
        "samples": chunk.tolist(),
        "cursor": state["cursor"],
        "total": total,
    }


@router.get("/status")
def get_status():
    return {
        "signals_loaded": list(_signals.keys()),
        "machine_states": _machine_states,
    }
