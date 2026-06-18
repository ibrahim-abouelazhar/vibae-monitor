import os
import sqlite3
from datetime import datetime

DB_PATH = "data/vibae_monitor.db"


def init_db():
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                machine_id     TEXT    NOT NULL,
                fault_type     TEXT,
                mse            REAL,
                threshold      REAL,
                is_anomaly     INTEGER,
                predicted_fault TEXT,
                severity_score REAL,
                rms            REAL,
                kurtosis       REAL,
                variance       REAL,
                fft_peak_hz    REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                machine_id     TEXT    NOT NULL,
                predicted_fault TEXT,
                mse            REAL,
                threshold      REAL,
                severity_score REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inf_machine ON inferences(machine_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inf_ts ON inferences(timestamp)")
        conn.commit()


def save_inference(machine_id: str, fault_type: str, mse: float, threshold: float,
                   is_anomaly: bool, predicted_fault: str, severity_score: float,
                   features: dict):
    ts = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO inferences
            (timestamp, machine_id, fault_type, mse, threshold, is_anomaly,
             predicted_fault, severity_score, rms, kurtosis, variance, fft_peak_hz)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts, machine_id, fault_type, mse, threshold, int(is_anomaly),
            predicted_fault, severity_score,
            features.get("rms", 0.0),
            features.get("kurtosis", 0.0),
            features.get("variance", 0.0),
            features.get("peak_frequency", 0.0),
        ))
        if is_anomaly:
            conn.execute("""
                INSERT INTO alerts
                (timestamp, machine_id, predicted_fault, mse, threshold, severity_score)
                VALUES (?,?,?,?,?,?)
            """, (ts, machine_id, predicted_fault, mse, threshold, severity_score))
        conn.commit()


def get_inferences(machine_id: str = None, limit: int = 200) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if machine_id:
            rows = conn.execute(
                "SELECT * FROM inferences WHERE machine_id=? ORDER BY id DESC LIMIT ?",
                (machine_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inferences ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_alerts(limit: int = 50) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_mse_series(machine_id: str, last_n: int = 50) -> list[float]:
    """Last N MSE values for a machine (chronological order)."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT mse FROM inferences WHERE machine_id=? ORDER BY id DESC LIMIT ?",
            (machine_id, last_n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]


def get_stats(machine_id: str = None) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        if machine_id:
            row = conn.execute("""
                SELECT COUNT(*) as total, SUM(is_anomaly) as anomalies,
                       AVG(mse) as avg_mse, MAX(severity_score) as max_sev,
                       AVG(severity_score) as avg_sev
                FROM inferences WHERE machine_id=?
            """, (machine_id,)).fetchone()
        else:
            row = conn.execute("""
                SELECT COUNT(*) as total, SUM(is_anomaly) as anomalies,
                       AVG(mse) as avg_mse, MAX(severity_score) as max_sev,
                       AVG(severity_score) as avg_sev
                FROM inferences
            """).fetchone()
        total = row[0] or 0
        anomalies = int(row[1] or 0)
        return {
            "total_inferences": total,
            "anomaly_count": anomalies,
            "anomaly_rate": round(anomalies / total, 4) if total else 0.0,
            "avg_mse": round(row[2] or 0.0, 6),
            "max_severity": round(row[3] or 0.0, 1),
            "avg_severity": round(row[4] or 0.0, 1),
        }


def get_per_machine_stats() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT machine_id, COUNT(*) as total, SUM(is_anomaly) as anomalies,
                   AVG(mse) as avg_mse, MAX(severity_score) as max_sev
            FROM inferences GROUP BY machine_id
        """).fetchall()
        return {
            r[0]: {
                "total": r[1], "anomalies": int(r[2] or 0),
                "avg_mse": round(r[3] or 0.0, 6),
                "max_severity": round(r[4] or 0.0, 1),
            }
            for r in rows
        }
