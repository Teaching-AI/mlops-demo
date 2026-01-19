from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris prediction API",
    description="API de description des espèces d'Iris",
    version="1.0.0"
)

# Charger le modèle au démarrage
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Modèle est chargé avec succès")
except Exception as e:
    logger.error(f"Error chargement modèle: {e}")
    model = None

# Schéma de validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Longeur sépale")
    sepal_width: float = Field(..., ge=0, le=10, description="Largeur sépale")
    petal_length: float = Field(..., ge=0, le=10, description="Longeur pétale")
    petal_width: float = Field(..., ge=0, le=10, description="Largeur pétale")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Route de santé
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Route de prédiction
@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="modèle non dispo")
    
    try:
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width            
        ]])

        # Prédiction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        logger.info(f"Prediction: {prediction} (confiance : {max(probabilities):.2f})")

        return {
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": {
                "setosa": float(probabilities[0]),
                "versicolor": float(probabilities[1]),
                "virginica": float(probabilities[2])
            }

        }

    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    