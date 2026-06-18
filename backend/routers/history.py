from fastapi import APIRouter, Query
from src.database import get_inferences, get_alerts, get_stats, get_per_machine_stats, get_mse_series

router = APIRouter()


@router.get("/inferences")
def list_inferences(
    machine_id: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Last N inferences (all machines or filtered by machine_id)."""
    return get_inferences(machine_id=machine_id, limit=limit)


@router.get("/alerts")
def list_alerts(limit: int = Query(50, ge=1, le=200)):
    """Last N anomaly alerts."""
    return get_alerts(limit=limit)


@router.get("/stats")
def global_stats(machine_id: str = Query(None)):
    """Aggregate statistics (optional filter by machine)."""
    return get_stats(machine_id=machine_id)


@router.get("/stats/machines")
def per_machine_stats():
    """Per-machine aggregate statistics."""
    return get_per_machine_stats()


@router.get("/mse-series")
def mse_series(
    machine_id: str = Query(...),
    last_n: int = Query(50, ge=5, le=200),
):
    """Chronological MSE series for a machine (for trend chart)."""
    series = get_mse_series(machine_id=machine_id, last_n=last_n)
    return {"machine_id": machine_id, "series": series, "count": len(series)}
