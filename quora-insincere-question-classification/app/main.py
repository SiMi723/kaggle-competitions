import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# Project root
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to find inference.py
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# Load inference pipeline
# --------------------------------------------------

from inference import predict_question

# --------------------------------------------------
# FastAPI application
# -------------------------------------------------

app = FastAPI(
    title="Quora Insincere Question Classifier",
    description="BiLSTM-based NLP classification API",
    version="1.0.0"
)
# --------------------------------------------------
# Request schema
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question text to classify"
    )

# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Quora Insincere Question Classifier API is running"
    }

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: QuestionRequest):

    try:
        prediction, probability = predict_question(
            request.question
        )

        label = (
            "Insincere"
            if prediction == 1
            else "Sincere"
        )

        return {
            "question": request.question,
            "prediction": prediction,
            "label": label,
            "probability": probability
        }

    except Exception:
         raise HTTPException(
                status_code=500,
                detail="Internal inference error"
            )
   