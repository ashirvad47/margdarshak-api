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

# --- Configuration Constants ---

# Use environment variables for deployment, with hardcoded fallbacks for easy local testing.
# Set these variables in your Vercel project settings.
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
# Vercel provides a writable /tmp directory
TEMP_DIR = Path("/tmp/models")

# Global variables to hold the loaded models
MODEL = None
LABEL_ENCODER = None
FEATURE_COLUMNS = None
models_loaded_successfully = False

# --- Pydantic Models for Input and Output ---

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

# *** FIXED RESPONSE MODEL ***
# This now matches what the Next.js frontend action expects to receive.
class PredictionOutput(BaseModel):
    top_predictions: List[CareerPrediction]

# --- FastAPI App Initialization ---
app = FastAPI(title="Career Prediction API")

# --- IMPORTANT: Add CORS Middleware ---
# This allows your Vercel frontend to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    # In production, replace "*" with your frontend's Vercel URL
    # e.g., allow_origins=["[https://your-app-name.vercel.app](https://your-app-name.vercel.app)"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions and Lifespan Events ---
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

@app.on_event("startup")
async def startup_event():
    """Load ML components when the application starts."""
    global MODEL, LABEL_ENCODER, FEATURE_COLUMNS, models_loaded_successfully
    
    print("--- FastAPI Startup Event: Loading ML Models ---")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    model_path = TEMP_DIR / "model.joblib"
    encoder_path = TEMP_DIR / "encoder.joblib"
    features_path = TEMP_DIR / "features.joblib"

    # Download all necessary files
    model_ok = download_file(MODEL_URL, model_path)
    encoder_ok = download_file(LABEL_ENCODER_URL, encoder_path)
    features_ok = download_file(FEATURE_COLUMNS_URL, features_path)
    
    if not (model_ok and encoder_ok and features_ok):
        print("FATAL: A required model component failed to download.")
        return # Models will remain None, and the flag will be False

    # Load models from files
    try:
        MODEL = joblib.load(model_path)
        LABEL_ENCODER = joblib.load(encoder_path)
        FEATURE_COLUMNS = joblib.load(features_path)
        models_loaded_successfully = True
        print("--- All ML components loaded successfully. API is ready. ---")
    except Exception as e:
        print(f"FATAL: Error loading models with joblib: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "models_loaded": models_loaded_successfully}

@app.post("/predict/", response_model=PredictionOutput)
def predict_career_api(payload: PredictionInput):
    """Receives candidate features, makes a prediction, and returns top 5 results."""
    if not models_loaded_successfully:
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