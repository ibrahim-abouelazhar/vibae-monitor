from fastapi import FastAPI  # REFACTOR
from fastapi.middleware.cors import CORSMiddleware  # REFACTOR

from backend.routers.chunk import router as chunk_router  # REFACTOR
from backend.routers.simulator import router as simulator_router, load_all_signals  # REFACTOR
from backend.routers.history import router as history_router  # REFACTOR
from backend.routers.batch_analysis import router as batch_router  # REFACTOR
from backend.routers.report import router as report_router  # REFACTOR
from src import model_service  # REFACTOR
from src.database import init_db  # REFACTOR

app = FastAPI(title="VibAE-Monitor API", description="API temps reel de detection et classification vibratoire")  # REFACTOR

app.add_middleware(  # REFACTOR
    CORSMiddleware,  # REFACTOR
    allow_origins=["*"],  # REFACTOR
    allow_credentials=True,  # REFACTOR
    allow_methods=["*"],  # REFACTOR
    allow_headers=["*"],  # REFACTOR
)  # REFACTOR


@app.on_event("startup")  # REFACTOR
def startup_event():  # REFACTOR
    if model_service.load_all():  # REFACTOR
        model, preprocessor, threshold_config = model_service.get_resources()  # REFACTOR
        classifier_model, classifier_config = model_service.get_classifier_resources()  # REFACTOR
        app.state.model = model  # REFACTOR
        app.state.preprocessor = preprocessor  # REFACTOR
        app.state.threshold_config = threshold_config  # REFACTOR
        app.state.classifier_model = classifier_model  # REFACTOR
        app.state.classifier_config = classifier_config  # REFACTOR
        app.state.buffers = {}  # REFACTOR
        print("Model resources loaded successfully.")  # REFACTOR
    else:  # REFACTOR
        app.state.buffers = {}  # REFACTOR
        print("WARNING: Model, scaler or threshold files not found.")  # REFACTOR
    load_all_signals()  # REFACTOR
    init_db()  # REFACTOR


@app.get("/health")  # REFACTOR
def health():  # REFACTOR
    model, _, threshold_config = model_service.get_resources()  # REFACTOR
    classifier_model, _ = model_service.get_classifier_resources()  # REFACTOR
    return {  # REFACTOR
        "status": "healthy" if model is not None else "waiting_for_model",  # REFACTOR
        "model_loaded": model is not None,  # REFACTOR
        "classifier_loaded": classifier_model is not None,  # REFACTOR
        "threshold": threshold_config["threshold"] if threshold_config else None,  # REFACTOR
        "model_type": threshold_config["model_type"] if threshold_config else None,  # REFACTOR
        "window_size": threshold_config["window_size"] if threshold_config else None,  # REFACTOR
    }  # REFACTOR


@app.get("/files")  # REFACTOR
def list_files():  # REFACTOR
    return {"message": "Mode temps reel actif. Envoyez un signal via POST /predict/chunk."}  # REFACTOR


app.include_router(chunk_router, prefix="/predict")  # REFACTOR
app.include_router(simulator_router, prefix="/simulate")  # REFACTOR
app.include_router(history_router, prefix="/history")  # REFACTOR
app.include_router(batch_router, prefix="/batch")  # REFACTOR
app.include_router(report_router, prefix="/report")  # REFACTOR
