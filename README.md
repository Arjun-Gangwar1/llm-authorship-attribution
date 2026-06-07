# LLM Authorship Attribution — 12-Class Text Classification

> Given a raw text string, identify which of **12 LLMs (or a human)** wrote it.
> Built for **CS204T: Artificial Intelligence** at IIT Dharwad as a 4-member team project.

---

## The Task

**Dataset:** [RAID](https://huggingface.co/datasets/liamdugan/raid) — 796,800 train / 49,800 val / 49,800 test rows, perfectly balanced across 12 classes.

**12 Classes:**
`gpt2` · `gpt3` · `gpt4` · `chatgpt` · `llama-chat` · `mistral` · `mistral-chat` · `mpt` · `mpt-chat` · `cohere` · `cohere-chat` · `human`

**Why it's hard:** Base vs chat-tuned variants of the same model (`mistral` vs `mistral-chat`, `mpt` vs `mpt-chat`, `cohere` vs `cohere-chat`) share vocabulary but differ in style. The `human` class has extreme length outliers (max 13,261 words vs avg 291).

---

## Results (Verified — Test Set, 49,800 rows)

| Model | Val Acc | Test Acc | Macro F1 | Train Time | Hardware |
|---|---|---|---|---|---|
| **RoBERTa-base (fine-tuned)** | **86.52%** | **86.51%** | **0.865** | ~9h | GPU (T4) |
| TF-IDF + Linear SVM | 78.29% | 78.40% | 0.782 | ~30 min | CPU only |
| TF-IDF + Logistic Regression | 76.96% | 76.73% | 0.766 | ~40 min | CPU only |
| E5-base + MLP | 59.73% | 59.61% | — | ~6h | GPU (embed) |
| TF-IDF + Naive Bayes | 43.65% | 43.34% | 0.414 | ~5 sec | CPU only |
| DeBERTa-v3-base | 8.29% | 8.29% | 0.013 | — | abandoned (NaN) |

*(All measured on the held-out test set, 796,800 train / 49,800 val / 49,800 test.)*

> **Key findings:**
> 1. **RoBERTa-base wins at 86.5%** — a fine-tuned transformer beats every classical method by 8+ points.
> 2. **TF-IDF + SVM is the best non-GPU model (78.4%)** — authorship signals are largely surface/lexical
>    (phrasing, punctuation, formatting), which TF-IDF captures directly and cheaply.
> 3. **Easiest classes:** gpt3 (F1 0.94), cohere (0.90). **Hardest:** the `*-chat` vs base pairs
>    (mistral-chat 0.80, cohere-chat 0.81) — a model and its chat-tuned variant write similarly.
> 4. **DeBERTa-v3 was abandoned** after NaN collapse across 3 configs — see "The Transformer Story" below.

---

## Quick Start

```bash
git clone https://github.com/ItzCobaltboy/llm-classifier
cd llm-classifier
pip install -r requirements.txt
```

**Run smoke test on demo data (no GPU, < 60 seconds):**
```bash
python scripts/01_smoke_test.py
```

**Train on full data** (requires RAID dataset download — see `data/README.md`):
```bash
python scripts/02_train_classical.py     # TF-IDF + SVM/LR/NB — best classical
python scripts/03_train_embeddings.py    # E5/BGE + MLP/LightGBM
python scripts/04_train_deberta.py       # DeBERTa fine-tune (GPU required)
```

---

## Project Structure

```
├── deberta_finetune_FIXED.py      ← DeBERTa fix (fp16 bug — see below)
├── llm-generated-text-2.ipynb     ← Main pipeline notebook (full outputs)
├── src/                           ← All source modules
│   ├── features/stylometric.py    ← 40 hand-crafted authorship features
│   ├── models/                    ← classical, deep, transformer trainers
│   └── evaluation/metrics.py      ← full metrics suite
├── TF-IDF/results/                ← Saved models + verified results JSON
├── llm_sbert_complete/            ← 46-run SBERT experiment grid + results
├── notebooks/                     ← Structured 10-notebook series
├── Final Codes/                   ← Ayush's TF-IDF+MLP (~80% F1)
├── my_messy_kitchen/cook_here/    ← 36-run DeBERTa grid + 3 production pipelines
├── Data_cleaning/                 ← 8 data-cleaning strategy scripts
├── deployment/                    ← Gradio demo + FastAPI endpoint
└── data/
    ├── demo/                      ← Small + medium subsets (pipeline testing)
    └── full/ → LLM_12_class/raid/ ← Real dataset (gitignored)
```

---

## The Transformer Story (Our Best Technical Narrative)

We first attempted **DeBERTa-v3-base**, which collapsed to **8.29% accuracy** (= random, 1/12 classes)
across **three separate configurations** on Kaggle T4 hardware:

| Config tried | Failure mode |
|---|---|
| fp16 | disentangled-attention softmax overflow → NaN gradients |
| fp32 + DataParallel (2× T4) | forward-pass gather corruption → 24,899/24,900 steps NaN |
| fp32 + single GPU + gradient_checkpointing | recomputation instability → 149,398 NaN batches |

We added every standard safeguard — finite-gradient guards, conservative LR, loss-NaN skipping,
fp32 precision — and DeBERTa-v3 *still* would not train stably on free hardware. Rather than burn
further compute chasing a fragile model, we made the engineering decision to switch to **RoBERTa-base**.

**RoBERTa trained cleanly on the first attempt** — stable fp16, smooth convergence, no NaN:

| Epoch | Train Loss | Val Loss | Val Acc |
|---|---|---|---|
| 1 | 0.749 | 0.535 | 81.39% |
| 2 | 0.402 | 0.431 | 85.45% |
| 3 | 0.279 | 0.423 | **86.52%** |

Final: **86.51% test accuracy, 0.865 macro-F1.** The lesson — *model stability on your actual hardware
matters more than picking the theoretically-best architecture* — is itself the most valuable takeaway.

Notebook: [`kaggle/kaggle_roberta.ipynb`](kaggle/kaggle_roberta.ipynb)

---

## Team

| Member | Contribution |
|---|---|
| Arjun Gangwar | EDA, TF-IDF pipeline, embedding caching (BGE/MiniLM/MPNet), transformer training (DeBERTa diagnosis + RoBERTa 86.5%), full Kaggle pipeline |
| Ayush Yadav | Data-cleaning suite (8 strategies), TF-IDF+MLP (~80% F1), 36-run DeBERTa grid search, RoBERTa/DistilBERT trainers, 3 production pipelines |
| Affan | Embedding pipeline (Qwen3/MiniLM/BERT/E5), DNN + CNN architectures, 16-run grid |
| Aaditya | 10-notebook structured series, DistilBERT HF Trainer, evaluate + predict scripts |

---

## Stack

PyTorch · HuggingFace Transformers · scikit-learn · LightGBM · sentence-transformers · Gradio · FastAPI · pandas · SHAP
