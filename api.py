from fastapi import FastAPI
from pydantic import BaseModel
from preprocess import TextPreprocessor
from model import ensemble_predict

app = FastAPI()

preprocessor = TextPreprocessor()

class Request(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "active"}


@app.post("/predict")
def predict(req: Request):

    text = preprocessor.normalize(req.text)

    # TODO:
    # - tokenize
    # - run GRU + Transformer
    # - ensemble output

    return {
        "text": text,
        "toxicity_scores": {
            "toxic": 0.0,
            "severe_toxic": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_hate": 0.0
        }
    }