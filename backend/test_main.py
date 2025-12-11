from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
from main import app  # Import app first

# ---------------------------------------------------------
# TESTS
# ---------------------------------------------------------

def test_read_root():
    """Verify the API is running."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_success():
    """Test /predict with valid payload and mocked models."""
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [45.2] 
    
    mock_nlp_results = {
        "topic": "Technology",
        "language": "en",
        "content_length": 50,
        "num_hashtags": 2,
        "sentiment_positive": 0.9,
        "sentiment_negative": 0.05,
        "sentiment_neutral": 0.05,
        "sentiment_category": "positive",
        "toxicity_score": 0.5
    }

    # PATCH targets in 'main' because that's where they are used.
    # main.py imports them from helpers/joblib, so we patch main.load_nlp_models, main.joblib.load
    with patch("main.joblib.load", return_value=mock_model), \
         patch("main.load_nlp_models", return_value={"mock": "pipeline"}), \
         patch("main.extract_caption_features", return_value=mock_nlp_results), \
         patch("main.predict_best_time_logic", return_value=(2, 14, 88.5)):
        
        # We must create TestClient INSIDE the patch context so lifespan runs with mocks
        with TestClient(app) as client:
            payload = {
                "platform": "Twitter",
                "caption": "Excited about AI! #tech",
                "followers": 1500,
                "account_age_days": 365,
                "verified": 1,
                "media_type": "Text",
                "location": "North America",
                "cross_platform_spread": 0
            }

            response = client.post("/predict", json=payload)
            
            # Debugging support
            if response.status_code != 200:
                print(response.json())

            assert response.status_code == 200
            data = response.json()
            
            assert data["best_day"] == 2
            assert data["best_hour"] == 14
            assert data["predicted_engagement"] == 88.5
            assert data["nlp_insights"]["topic"] == "Technology"

def test_predict_model_failure():
    """Test graceful 500 if models fail to load (simulating failure)."""
    # Here we don't patch, or we patch with side_effect=Exception
    # But strictly speaking, if we don't patch, it tries to load real models and fails -> 500.
    # Which is exactly what we want to test (that it catches exception or handles missing model).
    
    # However, since we can't easily "reset" the global var 'model' in a running app without 
    # dirty tricks, we can mock joblib to RAISE exception during lifespan.
    
    with patch("main.joblib.load", side_effect=Exception("Model missing")):
         with TestClient(app) as client:
            payload = {
                "platform": "Twitter",
                "caption": "Fail me",
                "followers": 100,
                "account_age_days": 100,
                "verified": 0,
                "media_type": "Text",
                "location": "Unknown",
                "cross_platform_spread": 0
            }
            
            response = client.post("/predict", json=payload)
            assert response.status_code == 500
