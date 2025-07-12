# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
from functools import lru_cache


# --- Pydantic Schemas for Request and Response ---

class InferenceRequest(BaseModel):
    text: str
    model_name: str  # e.g., 'bert-base-uncased', 'distilbert-base-uncased', etc.


class Prediction(BaseModel):
    label: str
    score: float


class InferenceResponse(BaseModel):
    predictions: list[Prediction]
    latency: float
    model_used: str


# --- Initialize FastAPI App ---
app = FastAPI()


# --- Model Loading with Caching ---
# This decorator caches the result of the function.
# When get_model() is called with the same model_name, the cached model is returned instantly.
@lru_cache(maxsize=3)  # Cache up to 3 models
def get_model(model_name: str):
    """Loads and caches a model and tokenizer."""
    print(f"Loading model: {model_name}...")
    # This path points to the final models saved by my advanced training script.
    model_path = f"./results/{model_name}-optimized/{model_name}-15epochs-final_model"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode
        print(f"Model {model_name} loaded successfully.")
        return model, tokenizer
    except OSError:
        print(f"Error: Model not found at path {model_path}")
        return None, None


# --- Label Information ---
# This list must match the labels my models were trained on.
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


# --- API Endpoint ---
@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    """Performs inference and measures latency."""

    # Start timer
    start_time = time.time()

    # Load model and tokenizer from cache
    model, tokenizer = get_model(request.model_name)

    if model is None or tokenizer is None:
        return InferenceResponse(predictions=[], latency=0, model_used=request.model_name)

    # Preprocess input text
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Post-process the output
    sigmoid = torch.sigmoid(logits)
    probabilities = sigmoid.cpu().numpy()[0]

    # Using a fixed threshold for demonstration.
    threshold = 0.8

    predictions = []
    for i, prob in enumerate(probabilities):
        if prob > threshold:
            predictions.append(Prediction(label=label_names[i], score=float(prob)))

    # End timer and calculate latency
    end_time = time.time()
    latency = end_time - start_time

    # Sort predictions by score for better readability
    predictions.sort(key=lambda x: x.score, reverse=True)

    return InferenceResponse(
        predictions=predictions,
        latency=latency,
        model_used=request.model_name
    )


@app.get("/")
def read_root():
    return {"status": "Emotion recognition API is running."}