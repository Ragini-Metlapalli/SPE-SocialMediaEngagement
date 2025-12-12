from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import re
from contextlib import asynccontextmanager



import os

os.environ["HF_HOME"] = "/models"
os.environ["TRANSFORMERS_CACHE"] = "/models"
os.environ["HF_CACHE"] = "/models"
os.environ["TORCH_HOME"] = "/models"



from helpers import (
    load_nlp_models,
    extract_caption_features,
    predict_best_time_logic
)

# ---------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------
# ---------------------------------------------------------
model = None
nlp_pipelines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from helpers import download_from_storj_if_missing
    # Load LightGBM/Sklearn Model
    download_from_storj_if_missing()
    global model
    try:
        model = joblib.load("best_time_model.pkl")
        print(" Main Model loaded successfully.")
    except Exception as e:
        print(f" Failed to load model: {e}")
        # We might want to exit here if critical, but for now we print error

    # Load Deep Learning Models
    global nlp_pipelines
    try:
        nlp_pipelines = load_nlp_models()
        print(" NLP Models loaded successfully.")
    except Exception as e:
        print(f" Failed to load NLP models: {e}")

    yield
    # Cleanup if needed
    nlp_pipelines.clear()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    platform: str
    followers: int
    account_age_days: int
    verified: int  # 0 or 1
    media_type: str
    location: str
    caption: str
    cross_platform_spread: int # 0 or 1

class PredictionResponse(BaseModel):
    best_day: int
    best_hour: int
    predicted_engagement: float
    nlp_insights: dict

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Social Media Engagement API is running with NLP Power"}



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Extract NLP Features
        nlp_features = extract_caption_features(request.caption, nlp_pipelines)

        # 2. Find Best Time
        best_day, best_hour, max_eng = predict_best_time_logic(
            model=model,
            req=request,
            nlp_features=nlp_features
        )

        return {
            "best_day": best_day,
            "best_hour": best_hour,
            "predicted_engagement": max_eng,
            "nlp_insights": nlp_features
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
