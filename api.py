from fastapi import FastAPI
from pydantic import BaseModel

from preprocess import TextPreprocessor
from model import load_best_model

app = FastAPI()

preprocessor = TextPreprocessor()

# Load best model once at startup
model, model_type = load_best_model()


class TextRequest(BaseModel):
    text: str


@app.get("/")
def health_check():
    return {
        "status": "active",
        "model_loaded": model_type
    }


@app.post("/predict")
def predict(req: TextRequest):

    cleaned_text = preprocessor.normalize(req.text)

    # -------------------------------------------------
    # TODO:
    # - tokenize input
    # - run inference using loaded model
    # -------------------------------------------------

    return {
        "input": cleaned_text,
        "model_used": model_type,
        "toxicity_scores": {
            "toxic": 0.0,
            "severe_toxic": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_hate": 0.0
        }
    }