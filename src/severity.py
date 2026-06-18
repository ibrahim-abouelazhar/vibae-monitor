from collections import deque

# Rolling MSE window per machine (in-memory, reset on server restart)
_mse_windows: dict[str, deque] = {}
_WINDOW_SIZE = 30  # number of inference windows kept for trend


def compute_severity(mse: float, threshold: float) -> float:
    """
    Map MSE / threshold ratio to a 0-100 severity score.

    0 – 30  : Healthy     (ratio ≤ 0.6)
    30 – 50 : Watch       (0.6 < ratio ≤ 1.0  → approaching threshold)
    50 – 75 : Anomaly     (1.0 < ratio ≤ 3.0)
    75 – 100: Critical    (ratio > 3.0)
    """
    if threshold <= 0:
        return 0.0
    ratio = mse / threshold
    if ratio <= 0.6:
        score = ratio / 0.6 * 30
    elif ratio <= 1.0:
        score = 30 + (ratio - 0.6) / 0.4 * 20
    elif ratio <= 3.0:
        score = 50 + (ratio - 1.0) / 2.0 * 25
    else:
        score = min(100.0, 75 + (ratio - 3.0) * 5)
    return round(score, 1)


def update_trend(machine_id: str, mse: float) -> dict:
    """
    Append new MSE value and return current trend for the machine.

    Returns:
        direction : "rising" | "stable" | "falling"
        slope     : linear-regression slope (raw)
        alert     : True if consistently rising toward threshold
        series    : current rolling MSE series
    """
    if machine_id not in _mse_windows:
        _mse_windows[machine_id] = deque(maxlen=_WINDOW_SIZE)
    _mse_windows[machine_id].append(mse)
    series = list(_mse_windows[machine_id])
    return {**_compute_trend(series), "series": series}


def _compute_trend(values: list[float]) -> dict:
    n = len(values)
    if n < 5:
        return {"direction": "stable", "slope": 0.0, "alert": False}

    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0

    # Normalise slope relative to current mean to get a scale-independent indicator
    ref = max(abs(y_mean), 1e-9)
    norm = slope / ref

    if norm > 0.03:
        direction = "rising"
    elif norm < -0.03:
        direction = "falling"
    else:
        direction = "stable"

    # Alert: rising AND last 5 values all above 70% of their window mean
    recent = values[-5:]
    alert = direction == "rising" and all(v > y_mean * 0.8 for v in recent)

    return {"direction": direction, "slope": round(slope, 8), "alert": alert}


def severity_label(score: float) -> str:
    if score < 30:
        return "Sain"
    if score < 50:
        return "Surveillance"
    if score < 75:
        return "Anomalie"
    return "Critique"


def severity_color(score: float) -> str:
    if score < 30:
        return "#00FF66"
    if score < 50:
        return "#FFB700"
    if score < 75:
        return "#FF6600"
    return "#FF3366"
