# File: api/index.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
import requests
import shutil
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydanticField
from pathlib import Path
import threading

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

# --- Globals for Models and a Lock for thread-safe loading ---
MODEL = None
LABEL_ENCODER = None
FEATURE_COLUMNS = None
model_loading_lock = threading.Lock()

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
    Communication_Skills: int = PydanticField(..., example=3, ge=0, le=5)
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

# --- Helper Functions and Model Loading ---
def download_file(url: str, destination: Path):
    print(f"Downloading from {url} to {destination}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"Download successful: {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def load_models():
    """
    Downloads and loads the ML models into the global variables.
    This function is designed to be called only when needed.
    """
    global MODEL, LABEL_ENCODER, FEATURE_COLUMNS
    
    # Ensure this runs only once
    with model_loading_lock:
        # Check again in case another thread finished loading while we were waiting
        if MODEL is not None:
            return

        print("--- First request: Loading ML Models ---")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        model_path = TEMP_DIR / "model.joblib"
        encoder_path = TEMP_DIR / "encoder.joblib"
        features_path = TEMP_DIR / "features.joblib"

        # Download all necessary files
        model_ok = download_file(MODEL_URL, model_path)
        encoder_ok = download_file(LABEL_ENCODER_URL, encoder_path)
        features_ok = download_file(FEATURE_COLUMNS_URL, features_path)
        
        if not (model_ok and encoder_ok and features_ok):
            print("FATAL: A required model component failed to download.")
            # Clear globals so the next request tries again
            MODEL, LABEL_ENCODER, FEATURE_COLUMNS = None, None, None
            raise HTTPException(status_code=503, detail="Model components failed to download.")

        # Load models from files
        try:
            MODEL = joblib.load(model_path)
            LABEL_ENCODER = joblib.load(encoder_path)
            FEATURE_COLUMNS = joblib.load(features_path)
            print("--- All ML components loaded successfully. API is ready. ---")
        except Exception as e:
            print(f"FATAL: Error loading models with joblib: {e}")
            # Clear globals so the next request tries again
            MODEL, LABEL_ENCODER, FEATURE_COLUMNS = None, None, None
            raise HTTPException(status_code=503, detail=f"Error loading models: {e}")


@app.get("/")
def read_root():
    # If models are loaded, status is "ready", otherwise "sleeping"
    status = "ready" if MODEL is not None else "sleeping"
    return {"status": status}

@app.post("/predict/", response_model=PredictionOutput)
def predict_career_api(payload: PredictionInput):
    """
    Receives features, ensures models are loaded, makes a prediction,
    and returns top 5 results.
    """
    # Load models only if they haven't been loaded yet
    if MODEL is None:
        load_models()
    
    # After load_models, if they are still None, it means loading failed.
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Models are not available. Please try again later.")

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