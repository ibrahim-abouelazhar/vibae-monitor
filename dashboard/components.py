"""Composants Plotly reutilisables pour le dashboard VibAE-Monitor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


BG_PANEL = "#1E293B"
TEXT = "#F1F5F9"
ACCENT = "#2E86AB"
ALERT = "#C0392B"
WARNING = "#D35400"
SUCCESS = "#1E8449"
ORANGE = "#F39C12"


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Applique le theme sombre industriel commun."""
    fig.update_layout(
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT, family="Inter, Roboto, sans-serif"),
        margin=dict(l=40, r=30, t=70, b=40),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(241, 245, 249, 0.15)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(241, 245, 249, 0.10)",
        zerolinecolor="rgba(241, 245, 249, 0.20)",
    )
    fig.update_yaxes(
        gridcolor="rgba(241, 245, 249, 0.10)",
        zerolinecolor="rgba(241, 245, 249, 0.20)",
    )
    return fig


def plot_signal_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mse: float,
    threshold: float,
) -> go.Figure:
    """Trace le signal original et sa reconstruction."""
    original = np.asarray(original, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    x = np.arange(len(original))
    is_fault = mse > threshold
    status = "ANOMALIE" if is_fault else "NORMAL"
    status_color = ALERT if is_fault else SUCCESS

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=original,
            mode="lines",
            name="Signal original",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=reconstructed,
            mode="lines",
            name="Signal reconstruit",
            line=dict(color=ORANGE, width=2),
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=ALERT,
        annotation_text=f"Seuil MSE: {threshold:.4f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(
            text=f"Reconstruction du signal | MSE={mse:.5f} | {status}",
            font=dict(color=status_color, size=18),
        ),
        xaxis_title="Point de la fenetre",
        yaxis_title="Amplitude normalisee",
        hovermode="x unified",
    )
    return _apply_dark_theme(fig)


def plot_mse_timeline(
    mse_scores: list[float],
    threshold: float,
    labels: list[str] | None = None,
) -> go.Figure:
    """Trace l'evolution des scores MSE par fenetre."""
    scores = np.asarray(mse_scores, dtype=float)
    x = np.arange(len(scores))
    anomalies = scores > threshold
    point_colors = np.where(anomalies, ALERT, SUCCESS)
    if labels is None or len(labels) != len(scores):
        labels = ["Non renseigne"] * len(scores)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scores,
            mode="lines",
            fill="tozeroy",
            name="MSE",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(46, 134, 171, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scores,
            mode="markers",
            name="Fenetre",
            marker=dict(color=point_colors, size=7, line=dict(color=TEXT, width=0.5)),
            customdata=np.asarray(labels, dtype=object),
            hovertemplate=(
                "Fenetre %{x}<br>"
                "MSE %{y:.5f}<br>"
                "Label reel %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=threshold,
        line_color=ALERT,
        line_width=2,
        annotation_text=f"Seuil: {threshold:.4f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Timeline des scores MSE",
        xaxis_title="Fenetre",
        yaxis_title="MSE",
        hovermode="closest",
    )
    return _apply_dark_theme(fig)


def plot_health_gauge(health_pct: float, machine_name: str) -> go.Figure:
    """Affiche une jauge de sante machine."""
    health_pct = float(np.clip(health_pct, 0, 100))
    if health_pct > 70:
        color = SUCCESS
    elif health_pct >= 40:
        color = WARNING
    else:
        color = ALERT

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_pct,
            number=dict(suffix="%", font=dict(color=color, size=34)),
            title=dict(text=f"Sante - {machine_name}", font=dict(color=TEXT, size=18)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT),
                bar=dict(color=color),
                bgcolor="rgba(241, 245, 249, 0.08)",
                borderwidth=1,
                bordercolor="rgba(241, 245, 249, 0.25)",
                steps=[
                    dict(range=[0, 40], color="rgba(192, 57, 43, 0.30)"),
                    dict(range=[40, 70], color="rgba(211, 84, 0, 0.30)"),
                    dict(range=[70, 100], color="rgba(30, 132, 73, 0.30)"),
                ],
                threshold=dict(
                    line=dict(color=TEXT, width=3),
                    thickness=0.75,
                    value=health_pct,
                ),
            ),
        )
    )
    fig.update_layout(height=320)
    return _apply_dark_theme(fig)


def plot_feature_comparison(
    df_normal_features: pd.DataFrame,
    df_fault_features: pd.DataFrame,
) -> go.Figure:
    """Compare RMS, Kurtosis et Variance entre normal et defaut."""
    features = ["RMS", "Kurtosis", "Variance"]
    missing_normal = [col for col in features if col not in df_normal_features.columns]
    missing_fault = [col for col in features if col not in df_fault_features.columns]
    if missing_normal or missing_fault:
        missing = sorted(set(missing_normal + missing_fault))
        raise ValueError(f"Colonnes manquantes pour le boxplot: {missing}")

    fig = go.Figure()
    for feature in features:
        fig.add_trace(
            go.Box(
                y=df_normal_features[feature],
                x=[feature] * len(df_normal_features),
                name="Normal",
                marker_color=SUCCESS,
                boxmean=True,
                legendgroup="Normal",
                showlegend=feature == features[0],
            )
        )
        fig.add_trace(
            go.Box(
                y=df_fault_features[feature],
                x=[feature] * len(df_fault_features),
                name="Defaut",
                marker_color=ALERT,
                boxmean=True,
                legendgroup="Defaut",
                showlegend=feature == features[0],
            )
        )

    fig.update_layout(
        title="Comparaison des features vibratoires",
        xaxis_title="Feature",
        yaxis_title="Valeur",
        boxmode="group",
    )
    return _apply_dark_theme(fig)
