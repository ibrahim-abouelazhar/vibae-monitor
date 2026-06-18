# VibAE-Monitor

Système de **maintenance prédictive en temps réel** pour roulements industriels, basé sur la détection d'anomalies vibratoires par autoencodeur convolutionnel et classification par DenseNet121.

---

## Architecture

```
VibAE-Monitor/
├── backend/
│   ├── main.py                  # FastAPI app, startup, CORS
│   └── routers/
│       ├── simulator.py         # Flux de signal CWRU en temps réel
│       ├── chunk.py             # /predict/chunk — inférence fenêtrée
│       ├── history.py           # /history — statistiques & alertes SQLite
│       ├── batch_analysis.py    # /batch — diagnostic complet par fichier
│       └── report.py            # /report/pdf — rapport PDF fpdf2
├── src/
│   ├── models/
│   │   ├── autoencoder.py       # CNN 2D autoencodeur (détection anomalie)
│   │   ├── classifier.py        # DenseNet121 fine-tuné (classification)
│   │   └── train.py             # Entraînement autoencodeur + DenseNet
│   ├── data_loader.py           # Préprocessing STFT → spectrogramme
│   ├── inference.py             # Pipeline d'inférence complet
│   ├── model_service.py         # Chargement des modèles
│   ├── severity.py              # Score sévérité 0–100 + tendance
│   └── database.py              # Persistance SQLite
├── frontend/
│   └── dashboard.html           # Dashboard temps réel (Plotly.js)
├── data/
│   └── processed/               # Modèles entraînés (non versionnés)
├── _archive/
│   └── data/raw/                # Fichiers .mat CWRU (non versionnés)
└── run.py                       # Lanceur unifié
```

---

## Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| **Détection anomalie** | Autoencodeur CNN 2D sur spectrogramme STFT — MSE vs seuil appris |
| **Classification défaut** | DenseNet121 fine-tuné (ImageNet → vibrations) — 10 classes CWRU |
| **Score sévérité** | 0–100 avec zones : Sain / Surveiller / Anomalie / Critique |
| **Tendance dégradation** | Régression linéaire sur fenêtre glissante MSE — alerte pente montante |
| **Temps réel** | Simulateur signal CWRU transparent (pas de sélection fichier côté utilisateur) |
| **Persistance** | SQLite — inférences, alertes, statistiques par machine |
| **Diagnostic batch** | Analyse complète d'un signal CWRU sur N fenêtres — précision, distribution |
| **Rapport PDF** | Export maintenance avec graphes MSE/sévérité et tableau d'alertes |

---

## Dataset

Projet basé sur le **CWRU Bearing Dataset** (Case Western Reserve University) :
- Fréquence d'échantillonnage : 48 kHz
- Accéléromètre Drive End (DE_time)
- 10 classes : Normal + 3 types × 3 sévérités (Ball, Inner Race, Outer Race)

> Les fichiers `.mat` ne sont pas inclus dans ce dépôt. Télécharger depuis :
> https://engineering.case.edu/bearingdatacenter/download-data-file

Placer les fichiers dans `_archive/data/raw/` :

| Fichier | Classe |
|---|---|
| `Time_Normal_1_098.mat` | Normal |
| `B007_1_123.mat` | Ball 0.007" |
| `B014_1_190.mat` | Ball 0.014" |
| `B021_1_227.mat` | Ball 0.021" |
| `IR007_1_110.mat` | Inner Race 0.007" |
| `IR014_1_175.mat` | Inner Race 0.014" |
| `IR021_1_214.mat` | Inner Race 0.021" |
| `OR007_6_1_136.mat` | Outer Race 0.007" |
| `OR014_6_1_202.mat` | Outer Race 0.014" |
| `OR021_6_1_239.mat` | Outer Race 0.021" |

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/<user>/vibae-monitor.git
cd vibae-monitor

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Installer les dépendances
pip install -r requirements.txt
```

---

## Entraînement des modèles

Les fichiers `.mat` CWRU doivent être présents dans `_archive/data/raw/`.

```bash
# 1. Autoencodeur (détection anomalie) — ~30 min CPU
python -m src.models.train --mode autoencoder

# 2. Classifieur DenseNet121 (progressive freezing) — ~2h CPU
python -m src.models.train --mode densenet
```

Les modèles sont sauvegardés dans `data/processed/`.

---

## Lancement

```bash
# Backend + Frontend
python run.py --all

# Backend seul
python run.py --backend

# Frontend seul
python run.py --frontend
```

| Service | URL |
|---|---|
| Backend API | http://127.0.0.1:8000 |
| Documentation API (Swagger) | http://127.0.0.1:8000/docs |
| Dashboard | http://127.0.0.1:8080/dashboard.html |

---

## API Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État des modèles chargés |
| GET | `/simulate/chunk?machine_id=&n=` | Chunk de signal simulé |
| POST | `/predict/chunk` | Inférence sur chunk accumulé |
| GET | `/history/stats` | Statistiques globales ou par machine |
| GET | `/history/inferences` | Dernières inférences SQLite |
| GET | `/history/alerts` | Dernières alertes anomalie |
| GET | `/history/mse-series` | Série temporelle MSE |
| GET | `/batch/run?fault_type=&max_windows=` | Diagnostic batch sur signal CWRU |
| GET | `/report/pdf` | Rapport PDF de maintenance |

---

## Modèles

### Autoencodeur CNN 2D
- Entrée : spectrogramme STFT (128×32, 1 canal)
- Architecture : Conv2D → BatchNorm → ReLU → ConvTranspose2D
- Entraîné sur signal **Normal** uniquement
- Détection anomalie : MSE reconstruction > seuil appris

### Classifieur DenseNet121
- Backbone DenseNet121 pré-entraîné ImageNet (poids `conv0` adaptés 3→1 canal)
- Progressive freezing : Phase 1 (backbone gelé, tête seule) → Phase 2 (denseblock4 dégelé)
- 10 classes de défauts CWRU
- ~92% accuracy

---

## Machines simulées

| ID | Nom |
|---|---|
| `machine_1` | Pompe P-01 |
| `machine_2` | Ventilateur V-02 |
| `machine_3` | Compresseur C-03 |
| `machine_4` | Convoyeur T-04 |

---

## Technologies

- **Backend** : FastAPI, Uvicorn, PyTorch, Torchvision, SciPy
- **Frontend** : HTML/CSS/JS vanilla, Plotly.js
- **Persistence** : SQLite (sqlite3 stdlib)
- **Rapport** : fpdf2, Matplotlib
- **Modèles** : CNN 2D Autoencoder, DenseNet121 Transfer Learning
