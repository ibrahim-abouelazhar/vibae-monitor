from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.chunk import router as chunk_router
from backend.routers.simulator import router as simulator_router, load_all_signals
from backend.routers.history import router as history_router
from backend.routers.batch_analysis import router as batch_router
from backend.routers.report import router as report_router
from src import model_service
from src.database import init_db

app = FastAPI(title="VibAE-Monitor API", description="API temps reel de detection et classification vibratoire")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    if model_service.load_all():
        model, preprocessor, threshold_config = model_service.get_resources()
        classifier_model, classifier_config = model_service.get_classifier_resources()
        app.state.model = model
        app.state.preprocessor = preprocessor
        app.state.threshold_config = threshold_config
        app.state.classifier_model = classifier_model
        app.state.classifier_config = classifier_config
        app.state.buffers = {}
        print("Model resources loaded successfully.")
    else:
        app.state.buffers = {}
        print("WARNING: Model, scaler or threshold files not found.")
    load_all_signals()
    init_db()


@app.get("/health")
def health():
    model, _, threshold_config = model_service.get_resources()
    classifier_model, _ = model_service.get_classifier_resources()
    return {
        "status": "healthy" if model is not None else "waiting_for_model",
        "model_loaded": model is not None,
        "classifier_loaded": classifier_model is not None,
        "threshold": threshold_config["threshold"] if threshold_config else None,
        "model_type": threshold_config["model_type"] if threshold_config else None,
        "window_size": threshold_config["window_size"] if threshold_config else None,
    }


@app.get("/files")
def list_files():
    return {"message": "Mode temps reel actif. Envoyez un signal via POST /predict/chunk."}


app.include_router(chunk_router, prefix="/predict")
app.include_router(simulator_router, prefix="/simulate")
app.include_router(history_router, prefix="/history")
app.include_router(batch_router, prefix="/batch")
app.include_router(report_router, prefix="/report")