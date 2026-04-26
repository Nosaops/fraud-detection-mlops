import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detects fraudulent transactions using a Random Forest model",
    version="1.0.0"
)

# Load the model once when the server starts
print("Loading fraud detection model...")
with open("gs_rf.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

# This defines exactly what the API expects to receive
class TransactionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# This defines what the API sends back
class PredictionResponse(BaseModel):
    prediction: int
    verdict: str
    fraud_probability: float
    legitimate_probability: float
    processing_time_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "model": "gs_rf.pkl", "ready": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    start_time = time.time()

    # Convert the incoming transaction into a numpy array
    # Order must match the training data column order
    features = np.array([[
        transaction.Time,
        transaction.V1,  transaction.V2,  transaction.V3,
        transaction.V4,  transaction.V5,  transaction.V6,
        transaction.V7,  transaction.V8,  transaction.V9,
        transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18,
        transaction.V19, transaction.V20, transaction.V21,
        transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Amount
    ]])

    # Get prediction and probability
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    processing_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        prediction=prediction,
        verdict="FRAUD" if prediction == 1 else "LEGITIMATE",
        fraud_probability=round(float(probabilities[1]), 4),
        legitimate_probability=round(float(probabilities[0]), 4),
        processing_time_ms=round(processing_ms, 2)
    )

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs": "Go to /docs to test the API visually"
    }
