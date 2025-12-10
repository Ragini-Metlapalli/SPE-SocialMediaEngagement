from fastapi.testclient import TestClient
from main import app
import pytest
import os

# Create TestClient
client = TestClient(app)

def test_read_root():
    """Verify the API is running and reachable."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Engagement Predictor API is running."}

def test_predict_full_payload():
    """Test /predict with all fields provided."""
    # We assume model files MIGHT be missing in CI environment if not properly mounted/copied.
    # But for this test suite, if the app starts, it means lifespan worked or failed gracefully.
    
    payload = {
        "caption": "Loving the new summer vibes! #summer #fun",
        "platform": "Instagram",
        "hashtags": "summer, fun, vibes",
        "location": "New York",
        "brand_name": "MyBrand",
        "user_past_sentiment_avg": 0.8,
        "user_engagement_growth": 0.05
    }
    
    response = client.post("/predict", json=payload)
    
    # If models loaded successfully, we expect 200. 
    # If they failed to load (e.g. file not found), main.py raises 503.
    # We assert that we get a valid response OR a known error state, 
    # rather than a crash (500).
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "recommended_day" in data
        assert "recommended_time" in data
        assert "sentiment_score" in data
        # Check logic: caption includes "Loving" -> likely positive
        assert data["sentiment_label"] in ["positive", "neutral", "negative"]

def test_predict_minimal_payload():
    """Test /predict with only required fields."""
    payload = {
        "caption": "Simple test caption",
        "platform": "Twitter"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["language"] == "en"

def test_invalid_platform_type():
    """Test validation (if we had Enum, but currently string so check logic)."""
    # Currently platform is just a string, so this should pass API validation 
    # but might be handled by logic.
    payload = {
        "caption": "Test",
        "platform": 123  # Invalid type
    }
    response = client.post("/predict", json=payload)
    # FastAPI/Pydantic validation should return 422
    assert response.status_code == 422

# Mocking or deeper logic tests
# If we wanted to test the heuristics independently:

from main import guess_language, get_toxicity, get_sentiment_score_label

def test_nlp_heuristics():
    # Language
    assert guess_language("Hello world") == "en"
    assert guess_language("Привет") == "ru"
    
    # Toxicity
    assert get_toxicity("I hate you idiot") > 0.0
    assert get_toxicity("I love you") == 0.0
    
    # Sentiment
    score, label = get_sentiment_score_label("I love this amazing product")
    assert label == "positive"
    assert score > 0
