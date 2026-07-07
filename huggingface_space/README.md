---
title: LLM Authorship Attribution
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
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
The Space loads the fine-tuned RoBERTa model
([ArjunGangwar1/llm-authorship-roberta](https://huggingface.co/ArjunGangwar1/llm-authorship-roberta))
from the Hugging Face Hub and runs inference on your input text, returning a
probability distribution over all 12 classes.

Code: [github.com/Arjun-Gangwar1/llm-authorship-attribution](https://github.com/Arjun-Gangwar1/llm-authorship-attribution)
