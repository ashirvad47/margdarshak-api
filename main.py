# File: api/index.py
import os
import joblib
import pandas as pd
import numpy as np
import requests
import shutil
import threading
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydanticField
from pathlib import Path

# --- Configuration Constants ---
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://career-pred.s3.ap-south-1.amazonaws.com/models/career_prediction_model_pipeline.joblib"
)
LABEL_ENCODER_URL = os.getenv(
    "LABEL_ENCODER_URL",
    "https://career-pred.s3.ap-south-1.amazonaws.com/models/career_label_encoder.joblib"
)
FEATURE_COLUMNS_URL = os.getenv(
    "FEATURE_COLUMNS_URL",
    "https://career-pred.s3.ap-south-1.amazonaws.com/models/career_feature_columns.joblib.joblib"
)
TEMP_DIR = Path("/tmp/models")

# --- Globals to Manage Model Loading State ---
# These variables will control the API's readiness.
models_state: Dict[str, any] = {
    "is_loading": False,
    "is_ready": False,
    "error": None
}
# A lock to prevent multiple threads from modifying the state at the same time.
state_lock = threading.Lock()

MODEL = None
LABEL_ENCODER = None
FEATURE_COLUMNS = None

# --- Pydantic Models ---
class PredictionFeatures(BaseModel):
    Field: str = PydanticField(..., example="B.Tech CSE")
    GPA: float = PydanticField(..., example=8.5)
    Leadership_Positions: int = PydanticField(..., example=1, ge=0)
    Research_Experience: int = PydanticField(..., example=0, ge=0)
    Industry_Certifications: int = PydanticField(..., example=1, ge=0)
    Extracurricular_Activities: int = PydanticField(..., example=2, ge=0)
    Internships: int = PydanticField(..., example=1, ge=0)
    Projects: int = PydanticField(..., example=3, ge=0)
    Field_Specific_Courses: int = PydanticField(..., example=4, ge=0)
    Coding_Skills: int = PydanticField(..., example=3, ge=0, le=5)
    Communication_Skills: int = PdanticField(..., example=3, ge=0, le=5)
    Problem_Solving_Skills: int = PydanticField(..., example=4, ge=0, le=5)
    Teamwork_Skills: int = PydanticField(..., example=3, ge=0, le=5)
    Analytical_Skills: int = PydanticField(..., example=3, ge=0, le=5)
    Presentation_Skills: int = PydanticField(..., example=2, ge=0, le=5)
    Networking_Skills: int = PydanticField(..., example=1, ge=0, le=5)

class PredictionInput(BaseModel):
    features: PredictionFeatures

class CareerPrediction(BaseModel):
    career: str
    probability: float

class PredictionOutput(BaseModel):
    top_predictions: List[CareerPrediction]

# --- FastAPI App Initialization ---
app = FastAPI(title="Career Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Background Model Loading ---
def download_file(url: str, destination: Path):
    print(f"Downloading {url}...")
    # Increased timeout for large files
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"Successfully downloaded {destination.name}")

def load_models_background():
    """
    This function runs in a separate thread. It downloads and loads the models,
    updating the global state dictionary upon completion or failure.
    """
    global MODEL, LABEL_ENCODER, FEATURE_COLUMNS
    
    with state_lock:
        models_state["is_loading"] = True

    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        model_path = TEMP_DIR / "model.joblib"
        encoder_path = TEMP_DIR / "encoder.joblib"
        features_path = TEMP_DIR / "features.joblib"

        # Download all files first
        download_file(MODEL_URL, model_path)
        download_file(LABEL_ENCODER_URL, encoder_path)
        download_file(FEATURE_COLUMNS_URL, features_path)
        
        print("All files downloaded. Loading into memory...")
        MODEL = joblib.load(model_path)
        LABEL_ENCODER = joblib.load(encoder_path)
        FEATURE_COLUMNS = joblib.load(features_path)
        print("--- All ML components loaded successfully. API is ready. ---")
        
        with state_lock:
            models_state["is_ready"] = True
            models_state["error"] = None

    except Exception as e:
        error_message = f"Failed to load models: {e}"
        print(error_message)
        with state_lock:
            models_state["error"] = error_message
    finally:
        with state_lock:
            models_state["is_loading"] = False


# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, create and start the background thread for model loading.
    The server does NOT wait for it to finish.
    """
    print("--- FastAPI starting up. Kicking off background model loading. ---")
    thread = threading.Thread(target=load_models_background)
    thread.start()


# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint to check the current status of the model loading."""
    return {"status": "ok", "model_status": models_state}

@app.post("/predict/", response_model=PredictionOutput)
def predict_career_api(payload: PredictionInput):
    """
    Receives features, checks model readiness, and returns predictions.
    """
    if models_state["is_loading"]:
        raise HTTPException(status_code=503, detail="Models are still being loaded. Please try again in a minute.")
    
    if models_state["error"]:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {models_state['error']}")
        
    if not models_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Models are not ready. Unknown state.")

    try:
        input_df = pd.DataFrame([payload.features.dict()])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        
        probabilities = MODEL.predict_proba(input_df)[0]
        top_n_indices = np.argsort(probabilities)[::-1][:5]
        
        top_predictions_list = [
            CareerPrediction(
                career=LABEL_ENCODER.classes_[i],
                probability=round(float(probabilities[i]), 4)
            ) for i in top_n_indices
        ]
        
        return PredictionOutput(top_predictions=top_predictions_list)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")