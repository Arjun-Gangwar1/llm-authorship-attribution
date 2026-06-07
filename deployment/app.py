"""Gradio demo app for LLM Text Classifier."""
import gradio as gr, pickle, numpy as np
from pathlib import Path
import sys; sys.path.insert(0,".")
from src.data.load_data import LE, CLASSES
from src.data.preprocess import clean_text
from src.features.stylometric import extract as stylo_extract

SAVE_DIR = Path("models/saved_models")
MODEL_PATH = SAVE_DIR / "best_model.pkl"  # update to your best model path
TFIDF_PATH = SAVE_DIR / "tfidf_vectorizers.pkl"

model, tw, tc, svd = None, None, None, None
try:
    with open(TFIDF_PATH,"rb") as f: tw, tc, svd = pickle.load(f)
    with open(MODEL_PATH,"rb") as f: model = pickle.load(f)
    print("✅ Model loaded")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")

def predict(text):
    if model is None: return {"error": "Model not loaded"}, ""
    text_c = clean_text(text,"light")
    from scipy.sparse import hstack
    X = hstack([tw.transform([text_c]), tc.transform([text_c])])
    pred = model.predict(X)[0]
    probs = {}
    if hasattr(model,"predict_proba"):
        p = model.predict_proba(X)[0]
        for cls, prob in sorted(zip(model.classes_, p), key=lambda x:-x[1])[:5]:
            probs[str(cls)] = float(prob)
    report = f"**Predicted: {pred}**\n\n" + "\n".join(f"- {k}: {v:.3f}" for k,v in probs.items())
    return probs, report

with gr.Blocks(title="LLM Text Classifier") as demo:
    gr.Markdown("# 🤖 LLM Text Classifier\n12-class authorship attribution")
    with gr.Row():
        txt = gr.Textbox(label="Input Text", lines=8, placeholder="Paste text here...")
        with gr.Column():
            out_label = gr.Label(label="Class Probabilities", num_top_classes=5)
            out_md    = gr.Markdown()
    btn = gr.Button("Predict 🚀")
    btn.click(predict, inputs=txt, outputs=[out_label, out_md])
    gr.Examples([
        ["Certainly! I'd be happy to help you with this comprehensive analysis."],
        ["ok so honestly idk what you mean tbh lol you know what i mean?"],
        ["The the the analysis is is is the the. And and and."],
    ], inputs=txt)

demo.launch(share=True)
