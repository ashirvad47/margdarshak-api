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
# Only the main model pipeline is needed now.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://career-pred.s3.ap-south-1.amazonaws.com/models/career_prediction_model_pipeline.joblib"
)
TEMP_DIR = Path("/tmp/models")


# --- Hardcoded Model Data ---
# By hardcoding these, we avoid two extra downloads on startup.

# 1. Feature Columns List (from your .joblib file)
FEATURE_COLUMNS = [
    'Field', 'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 
    'Leadership_Positions', 'Field_Specific_Courses', 'Research_Experience', 
    'Coding_Skills', 'Communication_Skills', 'Problem_Solving_Skills', 
    'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills', 
    'Networking_Skills', 'Industry_Certifications'
]

# 2. Career Classes List (from your LabelEncoder .joblib file)
CAREER_CLASSES = [
    'AI / Machine Learning Engineer', 'Actuarial Analyst', 'Advertising Manager', 
    'Aerospace Engineer', 'Analytical Chemist', 'Animator', 'Architect', 
    'Art Director', 'Biochemist', 'Biologist', 'Biomedical Engineer', 
    'Brand Manager', 'Business Analyst', 'Chartered Accountant', 
    'Chef / Culinary Artist', 'Chemical Engineer', 'Civil Engineer', 
    'Clinical Psychologist', 'Cloud Solutions Architect', 'Content Writer', 
    'Corporate Lawyer', 'Cost Accountant', 'Counseling Psychologist', 
    'Credit Analyst', 'Curator / Gallery Manager', 'Curriculum Developer', 
    'Cybersecurity Analyst', 'Data Center Engineer', 'Data Scientist', 'Dentist', 
    'DevOps Engineer', 'Digital Marketing Spec.', 'Doctor (MBBS)', 
    'Ecologist / Conservation Scientist', 'Education Administrator', 
    'Electrical Engineer', 'Electronics & Communication', 'Entrepreneur / Founder', 
    'Environmental Engineer', 'Environmental Scientist', 'Fashion Designer', 
    'Film / Video Editor', 'Financial Advisor', 'Financial Analyst', 
    'Financial Controller', 'Fine Artist / Painter', 'Geneticist', 
    'Graphic Designer', 'HR Manager', 'Hospitality Manager', 
    'Hotel Operations Manager', 'IT Project Manager', 'Illustrator', 
    'Industrial Engineer', 'Inorganic Chemist', 'Interior Designer', 
    'Investment Banker', 'Judge', 'Landscape Architect', 'Lawyer', 'Legal Consultant', 
    'Management Consultant', 'Market Research Analyst', 'Marketing Manager', 
    'Mathematician / Statistician', 'Mechanical Engineer', 
    'Medical Laboratory Technologist', 'Microbiologist', 'Mobile App Developer', 
    'Music Teacher', 'Music Therapist', 'Nuclear Physicist', 'Nurse', 
    'Nutritionist / Dietitian', 'Organic Chemist', 'Paralegal', 
    'Petroleum Engineer', 'Pharmacist', 'Physicist', 'Physiotherapist', 
    'Primary School Teacher', 'Public Health Specialist', 'Quantum Physicist', 
    'Radiographer / Imaging Technologist', 'Risk Analyst', 'School Counselor', 
    'School Principal', 'Secondary School Teacher', 'Social Media Manager', 
    'Social Worker', 'Software Developer', 'Sound Engineer', 
    'Special Education Teacher', 'Structural Engineer', 'Surgeon', 
    'Talent Acquisition Spec.', 'UX/UI Designer', 'University Professor', 
    'Urban Planner', 'Web Developer'
]


# --- Globals for Model Loading ---
models_state: Dict[str, any] = {"is_loading": False, "is_ready": False, "error": None}
state_lock = threading.Lock()
MODEL = None

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

# --- Background Model Loading ---
def load_model_background():
    global MODEL
    with state_lock:
        models_state["is_loading"] = True

    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        model_path = TEMP_DIR / "model.joblib"
        
        print(f"Downloading model from {MODEL_URL}...")
        with requests.get(MODEL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("Model downloaded. Loading into memory...")
        
        MODEL = joblib.load(model_path)
        print("--- Model loaded successfully. API is ready. ---")
        
        with state_lock:
            models_state["is_ready"] = True
    except Exception as e:
        error_message = f"Failed to load model: {e}"
        print(error_message)
        with state_lock:
            models_state["error"] = error_message
    finally:
        with state_lock:
            models_state["is_loading"] = False

# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    print("--- FastAPI starting up. Kicking off background model loading. ---")
    thread = threading.Thread(target=load_model_background)
    thread.start()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "model_status": models_state}

@app.post("/predict/", response_model=PredictionOutput)
def predict_career_api(payload: PredictionInput):
    if models_state["is_loading"]:
        raise HTTPException(status_code=503, detail="Model is still being loaded. Please try again in a minute.")
    if models_state["error"]:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {models_state['error']}")
    if not models_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model is not ready. Unknown state.")

    try:
        input_df = pd.DataFrame([payload.features.dict()])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        
        probabilities = MODEL.predict_proba(input_df)[0]
        top_n_indices = np.argsort(probabilities)[::-1][:5]
        
        top_predictions_list = [
            CareerPrediction(
                career=CAREER_CLASSES[i],  # Use the hardcoded list here
                probability=round(float(probabilities[i]), 4)
            ) for i in top_n_indices
        ]
        
        return PredictionOutput(top_predictions=top_predictions_list)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")