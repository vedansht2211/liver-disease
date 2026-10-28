# Import TestClient so we can simulate HTTP requests
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Reusable test client for all test functions
client = TestClient(app)


VALID_PAYLOAD = {
    "age": 62,
    "gender": "Male",
    "tot_bilirubin": 10.9,
    "direct_bilirubin": 5.5,
    "alkphos": 699,
    "sgpt": 64,
    "sgot": 100,
    "tot_proteins": 7.5,
    "albumin": 3.2,
    "ag_ratio": 0.74,
}


def test_health_endpoint():
    """Verify the application is running and responds to a health check."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_liver_prediction():
    """Verify valid lab values produce a prediction."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    result = response.json()

    # Response contract: all fields must exist
    assert "prediction" in result
    assert "class_name" in result
    assert "probability" in result
    assert "threshold" in result

    # Prediction must be 0 or 1, and the label must match it
    assert result["prediction"] in [0, 1]
    assert result["class_name"] in ["no liver disease", "liver disease"]
    assert result["class_name"] == ["no liver disease", "liver disease"][result["prediction"]]

    # Probability must be a valid probability
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_matches_threshold():
    """Verify the class label is consistent with probability and threshold."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    result = response.json()

    expected = int(result["probability"] >= result["threshold"])
    # Small tolerance: probability and threshold are rounded in the response
    if abs(result["probability"] - result["threshold"]) > 0.01:
        assert result["prediction"] == expected


def test_invalid_gender():
    """Verify an invalid gender value is rejected."""
    payload = {**VALID_PAYLOAD, "gender": "M"}
    response = client.post("/predict", json=payload)

    # FastAPI returns HTTP 422 for validation errors
    assert response.status_code == 422


def test_negative_bilirubin():
    """Verify negative lab values are rejected."""
    payload = {**VALID_PAYLOAD, "tot_bilirubin": -1.0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_field():
    """Verify a request with a missing field is rejected."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "albumin"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422