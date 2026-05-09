"""
Schémas Pydantic pour l'API VibAE-Monitor.
Définit les structures de requête et réponse de l'endpoint /predict.
"""
from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """Corps de la requête POST /predict.
    
    Envoyer une fenêtre de signal de exactement 1024 points
    normalisés ou bruts (la normalisation est faite côté API).
    """
    signal: list[float] = Field(
        ...,
        min_length=1024,
        max_length=1024,
        description="Fenêtre de signal vibratoire — exactement 1024 points float"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "signal": [0.01] * 1024
            }
        }
    }


class PredictResponse(BaseModel):
    """Réponse de l'endpoint /predict."""
    mse: float = Field(..., description="Erreur de reconstruction MSE")
    threshold: float = Field(..., description="Seuil µ+3σ calculé sur données normales")
    status: Literal["NORMAL", "ANOMALIE"] = Field(..., description="Décision finale")
    confidence: float = Field(..., description="MSE / threshold — ratio d'anomalie")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint /health."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    threshold: float | None


class RetrainResponse(BaseModel):
    """Réponse de l'endpoint /retrain."""
    status: str
    run_id: str
    metrics: dict[str, float]