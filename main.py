import os
import requests
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load trained model and scaler
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Enable CORS (Allows frontend on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (including localhost:8001)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Dummy metrics (Update this dynamically in model training)
latest_metrics = {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.85,
    "f1_score": 0.88
}


# Define Input Data Model
class PredictionInput(BaseModel):
    value: float


@app.get("/")
async def root():
    return {"message": "Server is running successfully!"}


@app.post("/predict")
async def predict(data: PredictionInput):
    """Predict if the given value is an anomaly or normal."""
    try:
        print(f"📥 Received Data: {data}")

        # Convert input to DataFrame
        df = pd.DataFrame([[data.value]])

        # Scale data
        X = scaler.transform(df)

        # Predict anomaly
        prediction = model.predict(X)
        print(f"✅ Prediction: {prediction}")

        return {"status": "Anomaly" if prediction[0] == -1 else "Normal"}

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Returns model performance metrics."""
    return latest_metrics


@app.get("/roc_curve")
async def get_roc_curve():
    """Serve the ROC curve image."""
    roc_path = "static/images/roc_curve.png"
    if os.path.exists(roc_path):
        return FileResponse(roc_path, media_type="image/png")
    return {"error": "ROC Curve image not found"}

@app.get("/confusion_matrix")
async def get_confusion_matrix():
    """Serve the confusion matrix image."""
    cm_path = "static/images/confusion_matrix.png"
    if os.path.exists(cm_path):
        return FileResponse(cm_path, media_type="image/png")
    return {"error": "Confusion matrix image not found"}



# Secure API key handling
API_KEY = os.getenv("IBM_API_KEY", "rEZVrJss_RizGpEq83sgycul6yV6m-Ipsdo7XfRgWW2_")
MODEL_URL = os.getenv("IBM_MODEL_URL", "https://us-south.ml.cloud.ibm.com")


def get_prediction(data):
    """Calls an external IBM Cloud ML Model."""
    if not API_KEY or not MODEL_URL:
        return {"error": "API Key or Model URL not configured"}

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(MODEL_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"External model error: {response.text}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
