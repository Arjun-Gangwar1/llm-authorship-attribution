"""FastAPI REST endpoint for LLM Text Classifier."""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np
from pathlib import Path
import sys; sys.path.insert(0,".")
from src.data.preprocess import clean_text

app = FastAPI(title="LLM Text Classifier API", version="1.0.0")
SAVE_DIR = Path("models/saved_models")

model = tw = tc = None
try:
    with open(SAVE_DIR/"tfidf_vectorizers.pkl","rb") as f: tw, tc, _ = pickle.load(f)
    with open(SAVE_DIR/"best_model.pkl","rb") as f: model = pickle.load(f)
except Exception as e:
    print(f"⚠️  {e}")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextInput):
    if model is None: return {"error": "Model not loaded"}
    from scipy.sparse import hstack
    text_c = clean_text(inp.text,"light")
    X = hstack([tw.transform([text_c]), tc.transform([text_c])])
    pred = model.predict(X)[0]
    result = {"prediction": str(pred)}
    if hasattr(model,"predict_proba"):
        probs = model.predict_proba(X)[0]
        result["probabilities"] = {str(c):float(p) for c,p in zip(model.classes_, probs)}
    return result

@app.get("/health")
def health(): return {"status": "ok"}

# Run: uvicorn deployment.api:app --reload
