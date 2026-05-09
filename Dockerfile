# ── VibAE-Monitor — Dockerfile ──────────────────────────────────────────────
# Image légère Python 3.11 slim
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Anass Aliate"
LABEL project="VibAE-Monitor"
LABEL description="API FastAPI — Détection d'anomalies vibratoires"

# Dossier de travail dans le container
WORKDIR /app

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements en premier (cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/      ./src/
COPY api/      ./api/
COPY models/   ./models/

# Exposer le port FastAPI
EXPOSE 8000

# Health check Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Lancement du serveur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
