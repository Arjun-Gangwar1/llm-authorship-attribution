# LLM Authorship Attribution — 12-Class Text Classification

> Given a raw text string, identify which of **12 LLMs (or a human)** wrote it.
> Fine-tuned RoBERTa reaches **86.5% test accuracy** on 796,800 training samples.

Built for **CS204T: Artificial Intelligence**, IIT Dharwad (team project — see [my role](#my-role-arjun-gangwar) and [acknowledgments](#acknowledgments)).

---

## The Task

**Dataset:** [RAID](https://huggingface.co/datasets/liamdugan/raid) — 796,800 train / 49,800 val / 49,800 test rows, perfectly balanced across 12 classes.

**12 Classes:**
`gpt2` · `gpt3` · `gpt4` · `chatgpt` · `llama-chat` · `mistral` · `mistral-chat` · `mpt` · `mpt-chat` · `cohere` · `cohere-chat` · `human`

**Why it's hard:** Base vs chat-tuned variants of the same model (`mistral` vs `mistral-chat`, `mpt` vs `mpt-chat`, `cohere` vs `cohere-chat`) share vocabulary but differ in style. The `human` class has extreme length outliers (max 13,261 words vs avg 291).

---

## Results (Verified — Held-out Test Set, 49,800 rows)

| Model | Val Acc | Test Acc | Macro F1 | Train Time | Hardware |
|---|---|---|---|---|---|
| **RoBERTa-base (fine-tuned)** | **86.52%** | **86.51%** | **0.865** | ~9h | GPU (T4) |
| TF-IDF + Linear SVM | 78.29% | 78.40% | 0.782 | ~30 min | CPU only |
| TF-IDF + Logistic Regression | 76.96% | 76.73% | 0.766 | ~40 min | CPU only |
| E5-base + MLP | 59.73% | 59.61% | — | ~6h | GPU (embed) |
| TF-IDF + Naive Bayes | 43.65% | 43.34% | 0.414 | ~5 sec | CPU only |
| DeBERTa-v3-base | 8.29% | 8.29% | 0.013 | — | abandoned (NaN) |

> **Key findings:**
> 1. **RoBERTa-base wins at 86.5%** — a fine-tuned transformer beats every classical method by 8+ points.
> 2. **TF-IDF + SVM is the best non-GPU model (78.4%)** — authorship signals are largely surface/lexical
>    (phrasing, punctuation, formatting), which TF-IDF captures directly and cheaply.
> 3. **Easiest classes:** gpt3 (F1 0.94), cohere (0.90). **Hardest:** the `*-chat` vs base pairs
>    (mistral-chat 0.80, cohere-chat 0.81) — a model and its chat-tuned variant write similarly.

---

## My Role (Arjun Gangwar)

This was a team project; my primary contributions were:

- **Exploratory Data Analysis** — class distributions, word-count outliers, the base-vs-chat confusion analysis ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)).
- **TF-IDF classical pipeline** — NaiveBayes / LogReg / Linear SVM; the SVM baseline reached **78.4%** ([`notebooks/04_main_pipeline.ipynb`](notebooks/04_main_pipeline.ipynb)).
- **The winning transformer** — fine-tuned **RoBERTa-base to 86.5%** test accuracy, the project's best result ([`notebooks/07_roberta_finetune.ipynb`](notebooks/07_roberta_finetune.ipynb)).
- **Transformer debugging** — diagnosed why DeBERTa-v3 collapsed to NaN across 3 GPU configs (see below) and made the call to switch to RoBERTa.
- **Embedding caching + full Kaggle pipeline** — BGE/MiniLM/MPNet caches and the end-to-end training/eval run on the full 796K dataset.

---

## The Transformer Story (My Best Technical Narrative)

I first attempted **DeBERTa-v3-base**, which collapsed to **8.29% accuracy** (= random, 1/12 classes)
across **three separate configurations** on Kaggle T4 hardware:

| Config tried | Failure mode |
|---|---|
| fp16 | disentangled-attention softmax overflow → NaN gradients |
| fp32 + DataParallel (2× T4) | forward-pass gather corruption → 24,899/24,900 steps NaN |
| fp32 + single GPU + gradient_checkpointing | recomputation instability → 149,398 NaN batches |

I added every standard safeguard — finite-gradient guards, conservative LR, loss-NaN skipping,
fp32 precision — and DeBERTa-v3 *still* would not train stably on free hardware. Rather than burn
further compute chasing a fragile model, I made the engineering decision to switch to **RoBERTa-base**.

**RoBERTa trained cleanly on the first attempt** — stable fp16, smooth convergence, no NaN:

| Epoch | Train Loss | Val Loss | Val Acc |
|---|---|---|---|
| 1 | 0.749 | 0.535 | 81.39% |
| 2 | 0.402 | 0.431 | 85.45% |
| 3 | 0.279 | 0.423 | **86.52%** |

Final: **86.51% test accuracy, 0.865 macro-F1.** The lesson — *model stability on your actual hardware
matters more than picking the theoretically-best architecture* — is the most valuable takeaway.

---

## Quick Start

```bash
git clone https://github.com/Arjun-Gangwar1/llm-authorship-attribution
cd llm-authorship-attribution
pip install -r requirements.txt
```

**Smoke-test the pipeline on the bundled demo data (no GPU, < 60 seconds):**
```bash
python scripts/01_smoke_test.py
```

**Train on the full dataset** (download RAID first — see [`data/README.md`](data/README.md)):
```bash
python scripts/pipeline2_tfidf_sklearn.py   # TF-IDF classical models (CPU)
python scripts/pipeline1_mpnet_svm.py       # sentence-embedding + SVM
# RoBERTa fine-tune (GPU): run notebooks/07_roberta_finetune.ipynb on Kaggle/Colab
```

---

## Project Structure

```
├── README.md, MASTER_PLAN.md, LICENSE
├── src/                    ← source package
│   ├── data/               ← unified loader, cleaning suite, preprocessing
│   ├── features/           ← stylometric (40 feats), TF-IDF, embeddings
│   ├── models/             ← classical, deep (DNN/CNN), DeBERTa fix
│   ├── training/           ← trainers + utilities
│   └── evaluation/         ← metrics + plots
├── notebooks/              ← EDA, main pipeline, RoBERTa, transformer experiments
├── scripts/                ← smoke test + training pipelines
├── deployment/             ← Gradio demo + FastAPI endpoint
├── huggingface_space/      ← live demo app (Hugging Face Space)
├── results/                ← results_all.json, master_results.csv, plots
├── config/config.yaml      ← configuration
└── data/demo/              ← small dataset for out-of-box smoke testing
```

---

## Stack

PyTorch · HuggingFace Transformers · scikit-learn · LightGBM · sentence-transformers · Gradio · FastAPI · pandas · SHAP

---

## Acknowledgments

Team project for CS204T (IIT Dharwad). Teammates contributed complementary components:
**Ayush Yadav** (data-cleaning suite, TF-IDF + MLP, DeBERTa/RoBERTa/DistilBERT grid scripts),
**Affan** (sentence-embedding pipeline, DNN/CNN architectures), and
**Aaditya** (DistilBERT HF Trainer, structured notebook scaffold).
The EDA, TF-IDF pipeline, and the winning RoBERTa fine-tune are my own work.
