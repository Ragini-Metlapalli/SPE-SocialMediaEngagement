from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest
import sys

# We need to mock 'helpers' and 'joblib' BEFORE importing main
# because main imports them at top level or in lifespan

# ---------------------------------------------------------
# MOCKS
# ---------------------------------------------------------
mock_model = MagicMock()
mock_model.predict.return_value = [45.2] # Mock predicted engagement

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

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
# Patching external dependencies to avoid loading heavy models
with patch("joblib.load", return_value=mock_model) as mock_load, \
     patch("helpers.load_nlp_models", return_value={"mock": "pipeline"}) as mock_nlp_load, \
     patch("helpers.extract_caption_features", return_value=mock_nlp_results) as mock_extract, \
     patch("helpers.predict_best_time_logic", return_value=(2, 14, 88.5)) as mock_logic:

    from main import app

    client = TestClient(app)

    # ---------------------------------------------------------
    # TESTS
    # ---------------------------------------------------------

    def test_read_root():
        """Verify the API is running."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_success():
        """Test /predict with valid payload and mocked models."""
        
        # We need to force the reload of dependencies inside the app context if needed,
        # but since we patched at import time, 'main.model' and 'main.nlp_pipelines' 
        # need to be manually set if the lifespan didn't run.
        # However, TestClient(app) runs the lifespan context manager!
        
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
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify schema
        assert "best_day" in data
        assert "best_hour" in data
        assert "predicted_engagement" in data
        assert "nlp_insights" in data
        
        # Verify values from our mocks
        assert data["best_day"] == 2
        assert data["best_hour"] == 14
        assert data["predicted_engagement"] == 88.5
        assert data["nlp_insights"]["topic"] == "Technology"

    def test_predict_missing_field():
        """Test validation error for missing field."""
        payload = {
            "platform": "Twitter",
            # missing caption
            "followers": 100
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_model_failure():
        """Test graceful 500 if models are missing (simulated)."""
        # Create a fresh app instance to simulate empty global state
        from main import app as fresh_app
        # We manually unset globals to simulate load failure
        import main
        main.model = None 
        main.nlp_pipelines = {}
        
        local_client = TestClient(fresh_app)
        
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
        
        response = local_client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Model not loaded" in response.json()["detail"]

