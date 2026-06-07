"""
LLM Authorship Attribution — Live Demo (Hugging Face Space)
============================================================
Paste any text → predicts which of 12 LLMs (or a human) wrote it.

Uses the fine-tuned RoBERTa-base model (86.5% test accuracy).
Runs on free HF Spaces CPU — first prediction warms up, then ~1-2s each.

The model weights are pulled from the Hugging Face Hub model repo at startup:
    HF_MODEL_REPO  (set below to your username/repo)
"""
import os
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CONFIG ───────────────────────────────────────────────────────────────────
# After you upload the model to the HF Hub (Step 3 in the guide), set this to
# "<your-username>/llm-authorship-roberta". Until then it falls back to a fresh
# roberta-base (which will give random predictions — upload your model first!).
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "roberta-base")

CLASSES = ['chatgpt', 'cohere', 'cohere-chat', 'gpt2', 'gpt3', 'gpt4',
           'human', 'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat']

DESCRIPTIONS = {
    'chatgpt':      'OpenAI ChatGPER — conversational, hedged, polished',
    'cohere':       'Cohere base — creative / long-form',
    'cohere-chat':  'Cohere chat-tuned — concise summaries',
    'gpt2':         'GPT-2 — Wikipedia-style factual prose',
    'gpt3':         'GPT-3 — short, direct',
    'gpt4':         'GPT-4 — structured, well-organized',
    'human':        'Human-written text',
    'llama-chat':   'LLaMA chat — narrative storytelling',
    'mistral':      'Mistral base — forum/Q&A style',
    'mistral-chat': 'Mistral chat — dense / academic',
    'mpt':          'MPT base — personal narrative',
    'mpt-chat':     'MPT chat — short, structured',
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
print(f"Loading model from: {HF_MODEL_REPO}  (device={DEVICE})")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_REPO, num_labels=len(CLASSES)
).to(DEVICE)
model.eval()
print("Model loaded.")


# ── PREDICT ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(text):
    if not text or not text.strip():
        return {}, "Please paste some text to classify."

    enc = tokenizer(text, max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEVICE)
    logits = model(input_ids=enc["input_ids"],
                   attention_mask=enc["attention_mask"]).logits
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Label -> probability dict for the gr.Label widget
    label_probs = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    top_idx = int(probs.argmax())
    top_cls = CLASSES[top_idx]
    report = (
        f"### Prediction: **{top_cls}**  ({probs[top_idx]*100:.1f}% confidence)\n\n"
        f"*{DESCRIPTIONS.get(top_cls, '')}*\n\n"
        "**Top 5 candidates:**\n"
        + "\n".join(
            f"- `{CLASSES[i]}` — {probs[i]*100:.1f}%"
            for i in probs.argsort()[::-1][:5]
        )
    )
    return label_probs, report


# ── UI ───────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Certainly! I'd be happy to help you with this comprehensive analysis. Here are the key points to consider, broken down systematically for clarity."],
    ["ok so honestly idk what you even mean by that lol. like, i get it but also not really? anyway whatever, it's fine i guess."],
    ["The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate."],
    ["In recent developments, the committee announced that the proposed measures would take effect next quarter, pending final approval from the regulatory authorities."],
]

with gr.Blocks(title="LLM Authorship Attribution", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🕵️ LLM Authorship Attribution\n"
        "**Which of 12 LLMs (or a human) wrote this text?**\n\n"
        "Fine-tuned RoBERTa-base · 86.5% test accuracy · trained on the RAID dataset (796,800 samples).\n"
        "Classes: `gpt2 · gpt3 · gpt4 · chatgpt · llama-chat · mistral · mistral-chat · "
        "mpt · mpt-chat · cohere · cohere-chat · human`"
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Input Text", lines=10,
                             placeholder="Paste any paragraph here...")
            btn = gr.Button("🔍 Identify the Author", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Class Probabilities", num_top_classes=5)
            out_md = gr.Markdown()
    btn.click(predict, inputs=txt, outputs=[out_label, out_md])
    txt.submit(predict, inputs=txt, outputs=[out_label, out_md])
    gr.Examples(EXAMPLES, inputs=txt)
    gr.Markdown(
        "---\n"
        "Built for CS204T (AI), IIT Dharwad · "
        "[GitHub repo](https://github.com/YOUR_USERNAME/llm-authorship-attribution) · "
        "Model: fine-tuned `roberta-base`"
    )

if __name__ == "__main__":
    demo.launch()
