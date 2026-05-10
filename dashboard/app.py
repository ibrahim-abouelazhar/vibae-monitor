"""Dashboard Streamlit VibAE-Monitor."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, MODELS_DIR, OVERLAP, WINDOW_SIZE  # noqa: E402
from src.data_loader import load_processed, normalize, sliding_window  # noqa: E402

from components import (  # noqa: E402
    ALERT,
    SUCCESS,
    TEXT,
    plot_health_gauge,
    plot_mse_timeline,
    plot_signal_reconstruction,
)


st.set_page_config(
    page_title="VibAE-Monitor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .stApp { background-color: #0F172A; color: #F1F5F9; }
    .stMetric { background-color: #1E293B; border-radius: 8px; padding: 12px; }
    .stButton>button { background-color: #2E86AB; color: white; border-radius: 6px; }
    html, body, [class*="css"] { font-family: Inter, Roboto, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    div[data-testid="stDataFrame"] { background-color: #1E293B; }
    .score-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid rgba(241, 245, 249, 0.10);
    }
    .score-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
    }
    .status-pill {
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0;
    }
  </style>""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialise les variables persistantes Streamlit."""
    defaults = {
        "alert_log": [],
        "analysis": None,
        "timeline_mse": None,
        "timeline_labels": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Charge les donnees traitees si elles sont disponibles."""
    try:
        df_train, df_test, _ = load_processed(DATA_PROCESSED, MODELS_DIR)
        return df_train, df_test
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return None, None


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Charge le scaler sauvegarde si disponible."""
    try:
        _, _, scaler = load_processed(DATA_PROCESSED, MODELS_DIR)
        return scaler
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, float]:
    """Charge les metriques du modele si le fichier existe."""
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_mse(window: np.ndarray) -> float:
    """Stub : remplacer par l'appel reel au modele P3."""
    # Simulation : MSE aleatoire ponderee, interchangeable avec detect.py plus tard.
    return float(np.random.uniform(0.01, 0.15))


def load_threshold() -> float:
    """Charge le seuil MSE sauvegarde ou retourne une valeur par defaut."""
    path = MODELS_DIR / "threshold.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return float(json.load(file)["threshold"])
    return 0.05


def is_anomaly(mse: float, threshold: float) -> bool:
    """Indique si le score MSE depasse le seuil."""
    return mse > threshold


def extract_signal_from_mat(uploaded_file) -> np.ndarray:
    """Extrait le signal Drive End d'un fichier .mat uploade."""
    uploaded_file.seek(0)
    mat_data = scipy.io.loadmat(uploaded_file)
    de_keys = [key for key in mat_data if not key.startswith("__") and key.endswith("_DE_time")]
    if not de_keys:
        available = [key for key in mat_data if not key.startswith("__")]
        raise ValueError(f"Aucune cle '*_DE_time' trouvee. Cles disponibles: {available}")
    return mat_data[de_keys[0]].flatten().astype(np.float32)


def build_uploaded_dataframe(uploaded_file, scaler) -> pd.DataFrame:
    """Transforme un fichier .mat uploade en DataFrame de fenetres."""
    signal = extract_signal_from_mat(uploaded_file)
    windows = sliding_window(signal, window_size=WINDOW_SIZE, overlap=OVERLAP)
    signal_columns = [f"window_{index}" for index in range(WINDOW_SIZE)]
    df = pd.DataFrame(windows, columns=signal_columns)
    df["label"] = "inconnu"
    df["source_file"] = uploaded_file.name

    if scaler is None:
        df_norm, _ = normalize(df)
        return df_norm

    df_norm, _ = normalize(df, scaler=scaler)
    return df_norm


def get_window(df: pd.DataFrame, index: int) -> np.ndarray:
    """Retourne une fenetre signal sous forme de vecteur NumPy."""
    return df.filter(like="window_").iloc[index].to_numpy(dtype=np.float32)


def simulate_reconstruction(window: np.ndarray, mse: float) -> np.ndarray:
    """Simule une reconstruction visuelle en attendant le modele P3."""
    noise = np.random.normal(0, 1, size=window.shape)
    noise_rms = np.sqrt(np.mean(noise**2))
    if noise_rms == 0:
        return window.copy()
    reconstructed = window + noise * (np.sqrt(mse) / noise_rms)
    return np.clip(reconstructed, 0, 1)


def compute_health(mse: float, threshold: float) -> float:
    """Convertit le MSE en score de sante 0-100."""
    if threshold <= 0:
        return 0.0
    return float(np.clip(100 * (1 - mse / (2 * threshold)), 0, 100))


def format_metric(metrics: dict[str, float], *keys: str) -> str:
    """Formate une metrique presente sous differents noms possibles."""
    for key in keys:
        if key in metrics:
            return f"{float(metrics[key]):.3f}"
    return "--"


def add_alert(source_file: str, mse: float, predicted: str, actual: str) -> None:
    """Ajoute une anomalie au journal d'alertes."""
    st.session_state["alert_log"].insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": source_file,
            "mse_score": round(float(mse), 6),
            "label_predit": predicted,
            "label_reel": actual,
        },
    )
    st.session_state["alert_log"] = st.session_state["alert_log"][:50]


def render_score_card(mse: float, threshold: float) -> None:
    """Affiche le score MSE courant avec une couleur de statut."""
    color = ALERT if is_anomaly(mse, threshold) else SUCCESS
    st.markdown(
        f"""
        <div class="score-card">
            <div style="color:{TEXT}; opacity:0.75;">Score MSE</div>
            <div class="score-value" style="color:{color};">{mse:.5f}</div>
            <div style="color:{TEXT}; opacity:0.75;">Seuil: {threshold:.5f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(status: str, anomaly: bool) -> None:
    """Affiche le statut de detection en grand."""
    color = ALERT if anomaly else SUCCESS
    background = "rgba(192, 57, 43, 0.22)" if anomaly else "rgba(30, 132, 73, 0.22)"
    st.markdown(
        f"""
        <div class="status-pill" style="background:{background}; color:{color};">
            {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(df: pd.DataFrame, window_index: int, threshold: float) -> dict:
    """Execute l'analyse d'une fenetre et retourne les resultats."""
    window = get_window(df, window_index)
    mse = predict_mse(window)
    anomaly = is_anomaly(mse, threshold)
    reconstructed = simulate_reconstruction(window, mse)
    label_predicted = "ANOMALIE" if anomaly else "NORMAL"
    label_real = str(df.iloc[window_index].get("label", "inconnu"))
    source_file = str(df.iloc[window_index].get("source_file", "signal_uploade"))

    if anomaly:
        add_alert(source_file, mse, label_predicted, label_real)

    return {
        "window": window,
        "reconstructed": reconstructed,
        "mse": mse,
        "anomaly": anomaly,
        "status": label_predicted,
        "label_real": label_real,
        "source_file": source_file,
        "health": compute_health(mse, threshold),
    }


init_state()
_, df_test = load_data()
scaler = load_scaler()
metrics = load_metrics()
threshold = load_threshold()
model_loaded = (MODELS_DIR / "autoencoder.pth").exists()

st.sidebar.title(":chart_with_upwards_trend: VibAE-Monitor")
st.sidebar.caption("Detection d'anomalies vibratoires CWRU")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier .mat", type=["mat"])

uploaded_df = None
if uploaded_file is not None:
    try:
        uploaded_df = build_uploaded_dataframe(uploaded_file, scaler)
        st.sidebar.success(f"{len(uploaded_df)} fenetres chargees")
    except ValueError as exc:
        st.sidebar.error(str(exc))

current_df = uploaded_df if uploaded_df is not None else df_test
current_source = "Fichier uploade" if uploaded_df is not None else "Dataset de test"

if current_df is not None and len(current_df) > 0:
    max_index = len(current_df) - 1
    selected_window = st.sidebar.slider(
        "Fenetre a analyser",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
    )
else:
    selected_window = 0
    st.sidebar.info("Aucune fenetre disponible pour l'instant.")

analyze_clicked = st.sidebar.button("Analyser", use_container_width=True)

if model_loaded:
    st.sidebar.success("Modele charge")
else:
    st.sidebar.warning("Modele non charge - stub actif")

if analyze_clicked and current_df is not None and len(current_df) > 0:
    st.session_state["analysis"] = run_analysis(current_df, selected_window, threshold)

st.title("VibAE-Monitor")
st.caption(f"Dashboard industriel de detection d'anomalies | Source active: {current_source}")

kpi_cols = st.columns(4)
kpi_cols[0].metric("F1-Score", format_metric(metrics, "f1_score", "f1", "F1-Score"))
kpi_cols[1].metric("ROC-AUC", format_metric(metrics, "roc_auc", "auc", "ROC-AUC"))
kpi_cols[2].metric("Recall", format_metric(metrics, "recall", "Recall"))
kpi_cols[3].metric("Alertes actives", len(st.session_state["alert_log"]))

analysis = st.session_state["analysis"]

left_col, right_col = st.columns([2, 1])
with left_col:
    st.subheader("Reconstruction")
    if analysis is not None:
        fig_reconstruction = plot_signal_reconstruction(
            analysis["window"],
            analysis["reconstructed"],
            analysis["mse"],
            threshold,
        )
        st.plotly_chart(fig_reconstruction, use_container_width=True)
        render_score_card(analysis["mse"], threshold)
    else:
        st.info("Chargez un signal pour lancer une analyse.")

with right_col:
    st.subheader("Sante machine")
    if analysis is not None:
        st.plotly_chart(
            plot_health_gauge(analysis["health"], "CWRU-DE"),
            use_container_width=True,
        )
        render_status(analysis["status"], analysis["anomaly"])
    else:
        st.markdown(
            f"<div class='score-card' style='color:{TEXT};'>En attente de signal.</div>",
            unsafe_allow_html=True,
        )

st.subheader("Timeline MSE")
if df_test is not None and len(df_test) > 0:
    if st.session_state["timeline_mse"] is None:
        timeline_df = df_test.head(200)
        st.session_state["timeline_mse"] = [
            predict_mse(get_window(timeline_df, index)) for index in range(len(timeline_df))
        ]
        st.session_state["timeline_labels"] = timeline_df["label"].astype(str).tolist()

    st.plotly_chart(
        plot_mse_timeline(
            st.session_state["timeline_mse"],
            threshold,
            st.session_state["timeline_labels"],
        ),
        use_container_width=True,
    )
else:
    st.info("Chargez un signal pour voir la timeline")

st.subheader("Journal des alertes")
if st.session_state["alert_log"]:
    st.dataframe(
        pd.DataFrame(st.session_state["alert_log"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.markdown(
        f"""
        <div class="score-card" style="color:{TEXT};">
            Aucune anomalie detectee dans cette session.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("A propos"):
    st.write(
        """
        VibAE-Monitor est un systeme de surveillance vibratoire base sur un
        autoencoder entraine sur les signaux CWRU. Le dashboard rassemble les
        indicateurs modele, la reconstruction du signal, la timeline MSE et le
        journal des alertes pour faciliter le diagnostic industriel.

        Projet realise par l'equipe VibAE-Monitor : data pipeline, modele,
        detection et visualisation Streamlit.
        """
    )
