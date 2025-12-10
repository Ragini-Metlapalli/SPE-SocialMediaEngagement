from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, CatBoostClassifier
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import logstash
from contextlib import asynccontextmanager

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# ----------------------------------------------------------
#                 LOGGING SETUP
# ----------------------------------------------------------
logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.INFO)
try:
   logger.addHandler(logstash.TCPLogstashHandler('logstash', 5000, version=1))
except:
   print("Logstash handler could not be added (Host not found?)")

# Global Model Storage (Loaded in lifespan for safety/mocking if needed, 
# but user script does top-level. We will do top-level but inside try/except to avoid crashes if files missing)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models directly from filesystem
    try:
        logger.info("Loading models...")
        
        models['main_model'] = CatBoostRegressor()
        models['main_model'].load_model("model.cbm")
        
        models['tfidf_general'] = joblib.load("tfidf_general.joblib")
        
        models['topic_model'] = joblib.load("topic_model.joblib")
        models['topic_vectorizer'] = joblib.load("topic_vectorizer.joblib")
        models['le_topic'] = joblib.load("le_topic.joblib") 
        
        models['emotion_model'] = joblib.load("emotion_model.joblib")
        models['emotion_vectorizer'] = joblib.load("emotion_vectorizer.joblib")
        models['le_emotion'] = joblib.load("le_emotion.joblib")
        
        if os.path.exists("label_encoders.joblib"):
             models['label_encoders'] = joblib.load("label_encoders.joblib")
        else:
             models['label_encoders'] = {}

        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        logger.error(f"Error loading models: {e}")

    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Engagement Predictor API is running."}

# ----------------------------------------------------------
#                   NLP HELPER FUNCTIONS
# ----------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def guess_language(text):
    text = text.lower()
    if re.search(r"[а-яё]", text): return "ru"
    if re.search(r"[一-龯ぁ-ゟ]", text): return "zh/ja"
    if re.search(r"[가-힣]", text): return "ko"
    if re.search(r"[ह-ॣ]", text): return "hi"
    if re.search(r"[ا-ي]", text): return "ar"
    return "en"

def get_sentiment_score_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.2:
        return score, "positive"
    elif score <= -0.2:
        return score, "negative"
    else:
        return score, "neutral"

TOXIC_WORDS = {"hate","stupid","idiot","trash","garbage","worst","sucks"}
def get_toxicity(text):
    t = text.lower()
    return sum(w in t for w in TOXIC_WORDS) / len(TOXIC_WORDS)

def get_keywords(text, top_n=5):
    if 'tfidf_general' not in models: return ""
    vec = models['tfidf_general'].transform([text])
    scores = vec.toarray()[0]
    if scores.sum() == 0:
        return ""
    feature_names = models['tfidf_general'].get_feature_names_out()
    top_idx = np.argsort(scores)[-top_n:][::-1]
    return " ".join(feature_names[i] for i in top_idx)

# ----------------------------------------------------------
#                     INPUT SCHEMA
# ----------------------------------------------------------
class UserInput(BaseModel):
    caption: str
    platform: str
    hashtags: Optional[str] = ""
    location: Optional[str] = "Unknown"
    brand_name: Optional[str] = "Unknown"
    product_name: Optional[str] = "Unknown"
    campaign_name: Optional[str] = "Unknown"
    campaign_phase: Optional[str] = "Unknown"
    user_past_sentiment_avg: float = 0.0
    user_engagement_growth: float = 0.0
    buzz_change_rate: float = 0.0

class ProcessingResult(BaseModel):
    language: str
    keywords: List[str]
    topic_categories: List[str]
    sentiment_score: float
    sentiment_label: str
    emotion_type: str
    toxicity_score: float
    recommended_day: str
    recommended_time: str

@app.post("/predict", response_model=ProcessingResult)
async def predict(data: UserInput):
    logger.info(f"Predicting for {data.platform}")
    if 'main_model' not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    text = data.caption + " " + (data.hashtags or "")

    # NLP extracted features
    lang = guess_language(text)
    keywords = get_keywords(text)
    sent_score, sent_label = get_sentiment_score_label(text)
    tox = get_toxicity(text)

    # Topic prediction
    X_topic = models['topic_vectorizer'].transform([text])
    topic_idx = models['topic_model'].predict(X_topic)[0]
    if hasattr(models['le_topic'], 'inverse_transform'):
         topic_label = models['le_topic'].inverse_transform([topic_idx])[0]
    else:
         topic_label = str(topic_idx)

    # Emotion prediction
    X_emotion = models['emotion_vectorizer'].transform([text])
    emo_idx = models['emotion_model'].predict(X_emotion)[0]
    if hasattr(models['le_emotion'], 'inverse_transform'):
        emotion_label = models['le_emotion'].inverse_transform([emo_idx])[0]
    else:
        emotion_label = str(emo_idx)

    # Build 168 combinations
    rows = []
    days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    
    for day in range(7):
        for hour in range(24):
            rows.append({
                "platform": data.platform,
                "location": data.location,
                "language_reg": lang,
                "topic_reg": topic_label,
                "sentiment_label_reg": sent_label,
                "emotion_reg": emotion_label,
                "brand_name": data.brand_name,
                "product_name": data.product_name,
                "campaign_name": data.campaign_name,
                "campaign_phase": data.campaign_phase,
                "sentiment_score_reg": sent_score,
                "toxicity_score_reg": tox,
                "keywords_reg": keywords,
                "keywords_len": len(keywords.split()),
                "hashtags_count": len(data.hashtags.split(",")) if data.hashtags else 0,
                "user_past_sentiment_avg": data.user_past_sentiment_avg,
                "user_engagement_growth": data.user_engagement_growth,
                "buzz_change_rate": data.buzz_change_rate,
                "day_num": day,
                "day_num_cat": str(day),
                "hour": hour
            })

    df = pd.DataFrame(rows)

    # Apply label encoders
    label_encoders = models.get('label_encoders', {})
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = 0

    # Only the needed features
    final_features = [
        "hour","sentiment_score_reg","toxicity_score_reg",
        "user_past_sentiment_avg","user_engagement_growth","buzz_change_rate",
        "keywords_len","hashtags_count"
    ] + list(label_encoders.keys())

    # Ensure all columns exist
    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    X = df[final_features]

    preds = models['main_model'].predict(X)
    df["pred"] = preds

    best_idx = np.argmax(preds)
    best_row = df.iloc[best_idx]
    
    best_day_str = days_map.get(int(best_row["day_num"]), "Monday")
    best_time_str = f"{int(best_row['hour']):02d}:00"

    return ProcessingResult(
        language=lang,
        keywords=keywords.split(),
        topic_categories=[topic_label],
        sentiment_score=float(sent_score),
        sentiment_label=sent_label,
        emotion_type=emotion_label,
        toxicity_score=float(tox),
        recommended_day=best_day_str,
        recommended_time=best_time_str
    )


