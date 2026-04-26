import pickle
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import your app
import sys
sys.path.insert(0, ".")
from fraudapp import app

client = TestClient(app)

# One real-looking transaction row
SAMPLE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
    "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
}

def test_health_endpoint():
    """API must return 200 and status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_200():
    """Prediction endpoint must return 200"""
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200

def test_predict_returns_verdict():
    """Response must contain verdict field"""
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    body = response.json()
    assert "verdict" in body
    assert body["verdict"] in ["FRAUD", "LEGITIMATE"]

def test_predict_returns_probability():
    """Fraud probability must be between 0 and 1"""
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert 0.0 <= body["legitimate_probability"] <= 1.0

def test_probabilities_sum_to_one():
    """Fraud + legitimate probabilities must sum to 1"""
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    body = response.json()
    total = body["fraud_probability"] + body["legitimate_probability"]
    assert abs(total - 1.0) < 0.001

def test_missing_field_returns_422():
    """Sending incomplete data must return validation error"""
    bad_request = {"Time": 0.0, "V1": -1.35}  # missing many fields
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422

def test_prediction_is_binary():
    """Prediction must be 0 or 1 only"""
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.json()["prediction"] in [0, 1]
