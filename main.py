# Render API: main.py
# This code acts as a secure proxy to your Hugging Face API.

import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- Environment Variables ---
# Get the Hugging Face Inference URL and your HF Token from Render's Environment Variables
HF_INFERENCE_URL = os.getenv("HF_INFERENCE_URL")
HF_API_TOKEN = os.getenv("HF_API_TOKEN") # This will be used for authorization

# --- CORS Middleware ---
# This allows your Vercel frontend to communicate with this Render API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: In production, change "*" to your actual Vercel app URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    # A simple endpoint to check if the proxy is running and configured
    return {
        "status": "ok",
        "proxy_configured": "true" if HF_INFERENCE_URL and HF_API_TOKEN else "false"
    }

# --- Proxy Prediction Endpoint ---
@app.post("/predict/")
async def proxy_predict(request: Request):
    # 1. Check if the service is configured
    if not HF_INFERENCE_URL or not HF_API_TOKEN:
        raise HTTPException(status_code=503, detail="Inference service is not configured on the server.")

    try:
        # 2. Get the JSON payload from the incoming request from your frontend
        payload = await request.json()

        # 3. Set up the authorization headers to securely access your private HF Space
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # 4. Forward the request to the Hugging Face Space
        response = requests.post(HF_INFERENCE_URL, headers=headers, json=payload)
        
        # 5. Raise an exception if the HF API returned an error (e.g., 4xx or 5xx)
        response.raise_for_status()

        # 6. Return the JSON response from the Hugging Face Space back to the frontend
        return response.json()

    except requests.exceptions.RequestException as e:
        # Handle network errors between Render and Hugging Face
        raise HTTPException(status_code=502, detail=f"Error communicating with inference service: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")