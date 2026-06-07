---
title: LLM Authorship Attribution
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# LLM Authorship Attribution — Live Demo

Paste any text and the model predicts which of **12 LLMs** (or a human) wrote it.

- **Model:** fine-tuned `roberta-base`
- **Accuracy:** 86.5% test accuracy (0.865 macro-F1)
- **Dataset:** RAID (796,800 training samples, 12 balanced classes)
- **Classes:** gpt2, gpt3, gpt4, chatgpt, llama-chat, mistral, mistral-chat, mpt, mpt-chat, cohere, cohere-chat, human

## How it works
The Space loads the fine-tuned RoBERTa model from the Hugging Face Hub and runs
inference on your input text, returning a probability distribution over all 12 classes.

Set the `HF_MODEL_REPO` Space variable to your model repo
(e.g. `your-username/llm-authorship-roberta`).

Built for CS204T (Artificial Intelligence), IIT Dharwad.
