import io
import base64
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from fastapi import APIRouter
from fastapi.responses import Response
from fpdf import FPDF

from src.database import get_stats, get_per_machine_stats, get_alerts, get_mse_series

router = APIRouter()

MACHINE_NAMES = {
    "machine_1": "Pompe",
    "machine_2": "Ventilateur",
    "machine_3": "Compresseur",
    "machine_4": "Convoyeur",
}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _chart_mse_trend(machine_id: str) -> bytes:
    series = get_mse_series(machine_id, last_n=50)
    if not series:
        series = [0.0]
    stats_g = get_stats()
    threshold = 0.031  # fallback; runtime value used in caption

    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    ax.plot(series, color="#00B8FF", linewidth=1.5, label="MSE")
    ax.axhline(threshold, color="#FF3366", linestyle="--", linewidth=1.2, label="Seuil")
    ax.set_xlabel("Fenêtre", color="#a0aec0", fontsize=8)
    ax.set_ylabel("MSE", color="#a0aec0", fontsize=8)
    ax.tick_params(colors="#a0aec0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    ax.legend(fontsize=7, facecolor="#1f2736", labelcolor="white", loc="upper left")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _chart_severity_bar(per_machine: dict) -> bytes:
    machines = list(per_machine.keys())
    severities = [per_machine[m].get("max_sev", 0) for m in machines]
    labels = [MACHINE_NAMES.get(m, m) for m in machines]
    colors = ["#00FF66" if s < 30 else "#FFB700" if s < 50 else "#FF6600" if s < 75 else "#FF3366"
              for s in severities]

    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(labels, severities, color=colors, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score sévérité", color="#a0aec0", fontsize=8)
    ax.tick_params(colors="#a0aec0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    for bar, val in zip(bars, severities):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── PDF builder ───────────────────────────────────────────────────────────────

def _build_pdf() -> bytes:
    global_stats = get_stats()
    per_machine  = get_per_machine_stats()
    alerts       = get_alerts(limit=20)
    generated_at = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title ──────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 200, 100)
    pdf.cell(0, 10, "VibAE-Monitor - Rapport de Maintenance Predictive", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Genere le : {generated_at}", ln=True, align="C")
    pdf.ln(4)

    # ── Global summary ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 184, 255)
    pdf.cell(0, 8, "Resume Global", ln=True)
    pdf.set_draw_color(0, 184, 255)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    summary = [
        ("Total d'inferences",   str(global_stats["total_inferences"])),
        ("Anomalies detectees",  f"{global_stats['anomaly_count']} ({global_stats['anomaly_rate']*100:.1f} %)"),
        ("MSE moyen global",     f"{global_stats['avg_mse']:.6f}"),
        ("Severite max observee",f"{global_stats['max_severity']:.1f} / 100"),
        ("Severite moyenne",     f"{global_stats['avg_severity']:.1f} / 100"),
    ]
    for label, value in summary:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, label + " :", ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, ln=True)
    pdf.ln(3)

    # ── Per-machine table ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 184, 255)
    pdf.cell(0, 8, "Statistiques par Machine", ln=True)
    pdf.set_draw_color(0, 184, 255)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)

    col_w = [40, 30, 30, 30, 35]
    headers = ["Machine", "Inferences", "Anomalies", "MSE Moy.", "Sev. Max"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 255)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for mid, s in per_machine.items():
        name = MACHINE_NAMES.get(mid, mid)
        sev = s.get("max_sev", 0)
        row = [name, str(s["total"]), str(s["anomalies"]),
               f"{s['avg_mse']:.5f}", f"{sev:.1f}"]
        fill = sev >= 75
        if fill:
            pdf.set_fill_color(255, 220, 220)
        for val, w in zip(row, col_w):
            pdf.cell(w, 6, val, border=1, fill=fill, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── Severity chart ─────────────────────────────────────────────────────────
    if per_machine:
        sev_png = _chart_severity_bar(per_machine)
        sev_b64 = base64.b64encode(sev_png).decode()
        tmp_path = "data/_tmp_sev.png"
        with open(tmp_path, "wb") as f:
            f.write(sev_png)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 184, 255)
        pdf.cell(0, 7, "Score de Severite par Machine", ln=True)
        pdf.image(tmp_path, x=30, w=140)
        pdf.ln(2)

    # ── MSE trend chart (machine_1) ────────────────────────────────────────────
    mse_png = _chart_mse_trend("machine_1")
    tmp_mse = "data/_tmp_mse.png"
    with open(tmp_mse, "wb") as f:
        f.write(mse_png)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 184, 255)
    pdf.cell(0, 7, "Evolution MSE - Pompe (machine_1)", ln=True)
    pdf.image(tmp_mse, x=15, w=170)
    pdf.ln(3)

    # ── Alert log ─────────────────────────────────────────────────────────────
    if alerts:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 51, 102)
        pdf.cell(0, 8, "Journal des Alertes (20 dernieres)", ln=True)
        pdf.set_draw_color(255, 51, 102)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)

        acol = [25, 35, 55, 30, 25, 25]
        ahdrs = ["#", "Heure", "Machine", "Diagnostic", "MSE", "Sev."]
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(255, 220, 220)
        for h, w in zip(ahdrs, acol):
            pdf.cell(w, 6, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 40, 40)
        for i, a in enumerate(alerts, 1):
            ts = a["timestamp"][:19].replace("T", " ")
            name = MACHINE_NAMES.get(a["machine_id"], a["machine_id"])
            row = [str(i), ts, name,
                   (a["predicted_fault"] or "")[:28],
                   f"{a['mse']:.5f}",
                   f"{a['severity_score']:.0f}"]
            for val, w in zip(row, acol):
                pdf.cell(w, 5, val, border=1, align="C")
            pdf.ln()

    return bytes(pdf.output())


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/pdf", response_class=Response)
def export_pdf():
    """Generate and download a maintenance report as PDF."""
    try:
        pdf_bytes = _build_pdf()
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(500, f"PDF generation failed: {e}")

    filename = f"vibae_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
