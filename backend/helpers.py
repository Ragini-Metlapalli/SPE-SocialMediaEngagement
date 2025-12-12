import pandas as pd
import numpy as np
import re
import subprocess
import os, glob
os.environ["HF_HOME"] = "/models"
os.environ["TRANSFORMERS_CACHE"] = "/models"
os.environ["HF_CACHE"] = "/models"
os.environ["TORCH_HOME"] = "/models"


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
TOPIC_OPTIONS = [
    "Finance", "Food", "Sports", "Education", "Gaming", "Climate",
    "Business", "Travel", "Fashion", "Politics", "Health",
    "Entertainment", "Science", "AI/ML", "Technology"
]



def download_from_storj_if_missing():
    target = "/models"

    # Check if models already exist in PVC
    if os.path.exists(f"{target}/models--facebook--bart-large-mnli"):
        print(" Models already in PVC. Skipping Storj download.")
        return

    print(" Downloading models from Storj to PVC...")

    ACCESS_GRANT = os.environ["STORJ_ACCESS_GRANT"]
    BUCKET = os.environ.get("STORJ_BUCKET", "hf-models")

    # Download entire folder from Storj
    cmd = f"""
    uplink --access '{ACCESS_GRANT}' cp -r sj://{BUCKET}/* {target}/
    """
    subprocess.run(cmd, shell=True, check=True)

    print(" Storj models downloaded.")


def resolve_snapshot(model_dir):
    snaps = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    return snaps[0]  # first snapshot directory

def load_nlp_models():
    download_from_storj_if_missing()


    from transformers import pipeline
    from detoxify import Detoxify
    """
    Loads heavy NLP models.
    NOTE: This might take time on first run.
    """
    # topic_classifier = pipeline(
    #     "zero-shot-classification",
    #     model="/models/models--facebook--bart-large-mnli",
    #     local_files_only=True
    # )

    # lang_detector = pipeline(
    #     "text-classification",
    #     model="/models/models--papluca--xlm-roberta-base-language-detection",
    #     local_files_only=True
    # )

    # sentiment_analyzer = pipeline(
    #     "sentiment-analysis",
    #     model="/models/models--cardiffnlp--twitter-xlm-roberta-base-sentiment",
    #     local_files_only=True
    # )


    base_dir = "/models"

    bart_dir = resolve_snapshot(f"{base_dir}/models--facebook--bart-large-mnli")
    topic_classifier = pipeline("zero-shot-classification", model=bart_dir)

    lang_dir = resolve_snapshot(f"{base_dir}/models--papluca--xlm-roberta-base-language-detection")
    lang_detector = pipeline("text-classification", model=lang_dir)

    sent_dir = resolve_snapshot(f"{base_dir}/models--cardiffnlp--twitter-xlm-roberta-base-sentiment")
    sentiment_analyzer = pipeline("sentiment-analysis", model=sent_dir)


    # toxicity_model = Detoxify(
    #     "original",
    #     checkpoint="/models/checkpoints/toxic_original-c1212f89.ckpt"
    # )

    os.environ["TORCH_HOME"] = "/models/checkpoints"
    toxicity_model = Detoxify("original")


    return {
        "topic": topic_classifier,
        "lang": lang_detector,
        "sentiment": sentiment_analyzer,
        "toxicity": toxicity_model
    }

def infer_topic(caption, classifier):
    result = classifier(caption, TOPIC_OPTIONS)
    return result["labels"][0]

def infer_language(caption, detector):
    result = detector(caption)[0]["label"]
    return result.lower()

def infer_sentiment(caption, analyzer):
    # return_all_scores=True gives us the full distribution
    results = analyzer(caption, return_all_scores=True)[0]
    
    # Example structure: [{'label': 'Negative', 'score': 0.01}, ...]
    prob_dict = { item["label"].lower(): float(item["score"]) for item in results }
    
    return {
        "pos": prob_dict.get("positive", 0.0),
        "neg": prob_dict.get("negative", 0.0),
        "neu": prob_dict.get("neutral", 0.0),
        "label": max(prob_dict, key=prob_dict.get)
    }

def infer_toxicity(caption, model):
    scores = model.predict(caption)
    return float(scores["toxicity"])

def extract_caption_features(caption, pipelines):
    """
    Runs all NLP models on the caption.
    """
    topic = infer_topic(caption, pipelines["topic"])
    language = infer_language(caption, pipelines["lang"])
    sentiment = infer_sentiment(caption, pipelines["sentiment"])
    toxicity = infer_toxicity(caption, pipelines["toxicity"])
    
    return {
        "topic": topic,
        "language": language,
        "content_length": len(caption),
        "num_hashtags": len(re.findall(r"#\w+", caption)),
        
        "sentiment_positive": sentiment["pos"],
        "sentiment_negative": sentiment["neg"],
        "sentiment_neutral": sentiment["neu"],
        "sentiment_category": sentiment["label"],
        
        "toxicity_score": toxicity * 100 # Scaling to 0-100 if model output is 0-1
    }

def predict_best_time_logic(model, req, nlp_features):
    """
    Generates a 7x24 grid and predicts engagement for each slot.
    Returns (Best Day, Best Hour, Predicted Engagement).
    """
    rows = []
    
    # We must match the EXACT feature order expected by the pipeline/model
    # Based on notebook analysis:
    # platform, followers, account_age_days, verified, media_type, location, 
    # topic, language, content_length, num_hashtags, 
    # sentiment_positive, sentiment_negative, sentiment_neutral, toxicity_score, 
    # day_of_week, hour_of_day, cross_platform_spread

    for day in range(7):
        for hour in range(24):
            row = {
                "platform": req.platform,
                "followers": req.followers,
                "account_age_days": req.account_age_days,
                "verified": req.verified,
                "media_type": req.media_type,
                "location": req.location,
                
                "topic": nlp_features["topic"],
                "language": nlp_features["language"],
                "content_length": nlp_features["content_length"],
                "num_hashtags": nlp_features["num_hashtags"],
                
                "sentiment_positive": nlp_features["sentiment_positive"],
                "sentiment_negative": nlp_features["sentiment_negative"],
                "sentiment_neutral": nlp_features["sentiment_neutral"],
                
                "toxicity_score": nlp_features["toxicity_score"],
                
                "day_of_week": day,
                "hour_of_day": hour,
                "cross_platform_spread": req.cross_platform_spread
            }
            rows.append(row)
            
    df_pred = pd.DataFrame(rows)
    
    # The pipeline in pickle file handles Preprocessing (OneHot) automatically
    predictions = model.predict(df_pred)
    
    df_pred["predicted_engagement"] = predictions
    
    # Find max
    best_row_idx = df_pred["predicted_engagement"].idxmax()
    best_row = df_pred.loc[best_row_idx]
    
    return (
        int(best_row["day_of_week"]),
        int(best_row["hour_of_day"]),
        float(best_row["predicted_engagement"])
    )
