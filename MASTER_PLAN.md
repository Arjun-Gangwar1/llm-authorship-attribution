# MASTER PLAN — LLM Authorship Attribution
## Complete Deep-Dive Analysis + Industry-Ready Project Guide

> **Every file and folder in this project has been read and analyzed.**
> This is the single source of truth. Read this before touching any code.
>
> **Data clarification (confirmed from screenshots + code):**
> - `Dataset/Dataset_small` + `Dataset/Dataset_medium` = **DEMO DATA** (pipeline smoke-tests only)
> - `LLM_12_class/raid/` = **THE REAL FULL DATASET** (run the final pipeline here)
>
> ✅ **CRITICAL ISSUE FIXED (2026-06-03):**
> `val.parquet` and `test.parquet` were identical — now fixed using Option B.
> Original val (99,600) split 50/50 stratified → new val (49,800) + new test (49,800).
> Backup of original at `LLM_12_class/raid/val/val_original_backup.parquet`.
> **All future results must be measured on this new val/test split.**
> Last updated: **2026-06-05**

---

## 🏆 FINAL RESULTS (measured on held-out test set, 49,800 rows)

| Rank | Model | Val Acc | Test Acc | Macro F1 | Hardware | Status |
|---|---|---|---|---|---|---|
| 🥇 | **RoBERTa-base (fine-tuned)** | **86.52%** | **86.51%** | **0.865** | T4 GPU, ~9h | ✅ BEST |
| 🥈 | TF-IDF + Linear SVM | 78.29% | 78.40% | 0.782 | CPU, ~30min | ✅ best classical |
| 🥉 | TF-IDF + LogReg | 76.96% | 76.73% | 0.766 | CPU | ✅ |
| 4 | E5-base + MLP | 59.73% | 59.61% | — | GPU embed | ✅ |
| 5 | TF-IDF + NaiveBayes | 43.65% | 43.34% | 0.414 | CPU, 5s | ✅ |
| — | DeBERTa-v3-base | 8.29% | 8.29% | 0.013 | — | ❌ abandoned (NaN ×3) |

**Headline: RoBERTa-base = 86.51% test accuracy** — beats best classical by 8 points.

**RoBERTa per-class F1 (test):** gpt3=0.94, cohere=0.90, gpt4=0.90, human=0.89, gpt2=0.87,
chatgpt=0.87, llama-chat=0.88, mpt=0.86, mpt-chat=0.84, mistral=0.83, cohere-chat=0.81, mistral-chat=0.80.
Hardest = the `*-chat` vs base pairs (as predicted).

**The transformer journey:** DeBERTa-v3 NaN-collapsed across 3 configs (fp16, DataParallel,
gradient_checkpointing) on T4 → switched to RoBERTa-base → trained cleanly first try
(81.39% → 85.45% → 86.52% over 3 epochs). Lesson: hardware-stability > theoretical-best architecture.

Notebook: `kaggle/kaggle_roberta.ipynb` · Model: `roberta_best.pt` · Results: `results/results_all.json`

---

## SECTION 1A — 🚨 CRITICAL: VAL == TEST (Read This First)

**Verified by direct `DataFrame.equals()` comparison:**
```python
val  = pd.read_parquet('LLM_12_class/raid/val/val.parquet')   # 99,600 rows
test = pd.read_parquet('LLM_12_class/raid/test/test.parquet') # 99,600 rows
val['generation'].equals(test['generation'])  # → TRUE
```

**What this means:**
- Every result in this project labeled "test accuracy" is actually **val accuracy measured twice**
- There is NO independent held-out test set on disk right now
- The canonical notebook (`llm-generated-text-2.ipynb`) reports val=test in the ensemble cell — it silently used the same file for both, which is why both numbers matched exactly (8.29% = 8.29%)
- This was likely caused during dataset download/preparation: whoever created `test.parquet` copied `val.parquet` instead of the real test split

**How to fix — 3 options (pick one):**

**Option A — Re-download the real test split (BEST)**
```python
# The RAID dataset on HuggingFace has a proper test split
from datasets import load_dataset
ds = load_dataset("liamdugan/raid", split="test")
ds.to_parquet("LLM_12_class/raid/test/test.parquet")
```

**Option B — Split the current val set in half (OK for now)**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
val = pd.read_parquet('LLM_12_class/raid/val/val.parquet')
new_val, new_test = train_test_split(val, test_size=0.5, stratify=val['model'], random_state=42)
new_val.to_parquet('LLM_12_class/raid/val/val.parquet', index=False)
new_test.to_parquet('LLM_12_class/raid/test/test.parquet', index=False)
# Result: ~49,800 rows each — still large enough for reliable evaluation
```

**Option C — Carve test out of train (acceptable)**
```python
train = pd.read_parquet('LLM_12_class/raid/train/train.parquet')  # 796,800 rows
# Take 10% of train as test (79,680 rows) — still leaves 717,120 for training
test = train.groupby('model').sample(frac=0.10, random_state=42)
train_new = train.drop(test.index)
test.to_parquet('LLM_12_class/raid/test/test.parquet', index=False)
train_new.to_parquet('LLM_12_class/raid/train/train.parquet', index=False)
```

**Recommendation: Use Option B immediately** (takes 10 seconds, no download needed).
Then report honestly in the README: "Test set = 50% of original val split (49,800 rows), stratified."

**Impact on existing results:**
- All "val accuracy" numbers (78.71%, 60.88%, etc.) remain valid — they were measured on the real val set
- All "test accuracy" numbers are meaningless duplicates — do not report them
- After fixing, run `scripts/06_final_evaluation.py` once to get real test numbers

---

## SECTION 1 — THE ACTUAL DATASET (What's Really On Disk)

### Real Full Dataset — `LLM_12_class/raid/`
```
train/train.parquet              →  796,800 rows  (spec said 960k — 83% present)
val/val.parquet                  →   49,800 rows  (✅ fixed 2026-06-03, was 99,600 = duplicate of test)
test/test.parquet                →   49,800 rows  (✅ fixed 2026-06-03, was duplicate of val)
val/val_original_backup.parquet  →   99,600 rows  (backup of original before fix)
Columns: model (label), generation (text)
```

### The 12 Classes (verified from data):
```
chatgpt, cohere, cohere-chat, gpt2, gpt3, gpt4,
human, llama-chat, mistral, mistral-chat, mpt, mpt-chat
```

### Class Balance (verified — truly balanced):
Each class has exactly ~66,300–66,500 rows in train (±200 rows variation = perfectly balanced).

### Word Count Per Class (verified from actual full train data — 796,800 rows):
| Class | Avg Words | Min | Max | Std Dev | Notes |
|---|---|---|---|---|---|
| chatgpt | 260 | 1 | 902 | 109 | conversational, moderate length |
| cohere | 230 | 3 | 705 | 104 | |
| cohere-chat | 185 | 1 | 554 | 86 | shorter than base cohere |
| gpt2 | 291 | 1 | 704 | 115 | Wikipedia-style articles |
| gpt3 | 134 | 1 | 637 | 75 | **shortest avg** — brief reviews |
| gpt4 | 271 | 3 | 1574 | 92 | book/content summaries |
| **human** | **291** | **1** | **13,261** | **311** | ⚠️ huge outliers (std=311, max=13k words) |
| llama-chat | 266 | 28 | 1293 | 65 | narrative style |
| mistral | 253 | 1 | 783 | 97 | forum/Q&A with headers |
| mistral-chat | 200 | 1 | 786 | 75 | dense academic |
| mpt | 276 | 1 | 975 | 121 | personal narratives |
| mpt-chat | 163 | 1 | 905 | 87 | **second shortest** |

**Critical insight from this table:**
- `human` class has extreme outliers: max=13,261 words (std=311 vs avg=291). This means some human texts are 40× longer than average. Ayush's data cleaning (truncate to 500 words) is the right call.
- `gpt3` is the shortest class (avg 134 words) — this length alone is a discriminating feature.
- The `*-chat` variants are consistently shorter than their base models: gpt3-like brevity for chat-tuned models.

**Key insight from actual text samples:**
- `chatgpt`: Conversational, hedged opinions ("I've been thinking a lot about…")
- `cohere`: Creative writing style, sometimes poetic
- `cohere-chat`: News summarization style
- `gpt2`: Wikipedia-style factual articles (longest texts)
- `gpt3`: Short, direct review-style text
- `gpt4`: Book/content summaries, well-structured
- `human`: News articles, BBC-style
- `llama-chat`: Narrative storytelling
- `mistral`: Forum/Reddit Q&A style with headers
- `mistral-chat`: Dense academic/technical writing
- `mpt`: Personal narrative/advice-seeking
- `mpt-chat`: Short, structured

**The hard classification pairs:** `cohere` vs `cohere-chat`, `mpt` vs `mpt-chat`, `mistral` vs `mistral-chat` — base vs chat-tuned variants of the same model write very differently but share vocabulary.

⚠️ **Unicode issue detected:** Some texts contain `​` (zero-width space) and non-ASCII chars — need UTF-8 encoding everywhere.

### Demo Data — `Dataset/Dataset_small` and `Dataset/Dataset_medium`
```
Dataset_small:  train ~7,968 rows  (~664/class)   cols: text, generated_by
Dataset_medium: train 79,681 rows  (~6,640/class)  cols: text, generated_by
```
⚠️ Different column names: `text` and `generated_by` (not `model`, `generation`). Must rename on load.
These are used ONLY for smoke-testing pipelines before running on the real data.

---

## SECTION 2 — COMPLETE FILE-BY-FILE ANALYSIS

### 2A. Arjun's Work (canonical pipeline)

**`llm-generated-text-2.ipynb`** — The canonical 10-cell notebook, fully executed on Kaggle (2× T4 GPUs):
- Cell 1: Install packages
- Cell 2: GPU setup — detected 2× Tesla T4 (14.6 GB each)
- Cell 3: CONFIG + save/load helpers
- Cell 4: Load medium split (79,681 train / 9,959 val / 9,959 test), column normalisation
- Cell 5: EDA — class distribution, word count boxplot
- Cell 6: **TF-IDF baselines** (80k features, 1-2 ngrams, sublinear_tf): NB=41.53%, LR=62.64%, SVM=62.70%
- Cell 7: **SBERT + Stylometric + LightGBM**: MiniLM(38.4%), MPNet(41.1%), BGE-base(44.1%), then LightGBM on BGE=60.88%
- Cell 8: **DeBERTa training** — ALL 5 BUGS IDENTIFIED AND FIXED in the code comments (fp16, checkpointing+DataParallel, LR, max_length, GradScaler incompatibility)
- Cell 9: Ensemble (DeBERTa 65% + LGBM 25% + SVM 10%)
- Cell 10: Final results bar chart

**`TF-IDF/results/models/`** — Saved artifacts:
- `full_tfidf_svm.pkl` — **best model** (78.71% val)
- `full_tfidf_vectorizer.pkl`, `full_stylo_scaler.pkl`, `label_encoder.pkl`
- `mini_*.pkl`, `mini_deberta_best.pt`
- `results_all.json` — the verified results ledger
- `emb_cache/` — BGE/MiniLM/MPNet `.npy` caches (full + mini, train + val)
- `hf_cache/` — downloaded DeBERTa-v3-base weights

**`deberta_finetune_FIXED.py`** — The standalone DeBERTa fix. 248 lines. All 5 bugs addressed:
- Bug A: gradient_checkpointing disabled with DataParallel
- Bug B: LR 1e-5 encoder, 3× head (not 2e-5/10×)
- Bug C: `_finite_grads()` — checks gradients, not just loss
- Bug D: max_length=256 (not 512)
- Bug E: FP32 forced when DataParallel (GradScaler.unscale_() incompatible)

**`EDA/EDA.ipynb`** — Arjun's EDA notebook (class distribution, word counts, vocab richness, punctuation patterns, duplicates, outliers). Produced `eda_0*.png` plots.

---

### 2B. Affan's Work (embedding pipeline + DNN/CNN)

**`clean.py`** — Truncates text to 300 words (normalises length — human texts are much longer), then downsamples each class to the smallest class count for perfect balance. Operates on `Dataset/Dataset_small` and `Dataset/Dataset_medium` only (the demo data).

**`explore_data.py`** — Word count stats per class per split. Shows class distribution and min/max/avg word counts.

**`embedding_models.py`** — Demo/test script that loads and tests all 4 embedding models:
- Qwen3-Embedding-0.6B (with query prompt)
- all-MiniLM-L6-v2
- BERT base (via fill-mask pipeline — **note: this is wrong for embeddings**, should be mean-pooled)
- E5-base (with "passage:" prefix)

**`embed.py`** — Production embedding generator. Loads all 4 models, runs on `Dataset_small` (currently `SIZES=["small"]`), saves `.pt` files under `embeddings/`. Also handles BPE-style padding to nearest multiple of 8. Correctly uses "passage: " prefix for E5.

**`models.py`** — Two PyTorch architectures:
- `DNN`: `input_dim → [512 → 256 → 128] → 12` with ReLU + Dropout(0.3)
- `CNN`: 1D conv over embedding vector as signal: `[Conv(1,32) → Conv(32,64) → Conv(64,128) → AdaptiveAvgPool(64)] → Flatten → [8192 → 256 → 12]`
Both expose `.fit()` and `.evaluate()` via a `Classifier` wrapper.

**`train.py`** — 16-run grid: `4 embeddings × 2 architectures × 2 dataset sizes`. Loads `.pt` files from `embeddings/`. Reports classification_report on test set. **NOT YET RUN** — `embeddings/` directory doesn't exist yet (embed.py must be run first).

**`pyproject.toml` + `uv.lock`** — Modern `uv` package management (faster than pip). Dependencies: torch, sentence-transformers, transformers, scikit-learn, pandas, tqdm.

---

### 2C. Ayush's Work (data cleaning + TF-IDF+NN + DeBERTa grid)

**`Data_cleaning/` (8 scripts + utils.py):**
- `utils.py` — core functions: `remove_outliers()` (IQR), `truncate()`, `pad_and_truncate()`, `split_text()`, `upsample()`, `downsample()`
- `remove_outliers_downsample.py` — remove IQR outliers + downsample
- `remove_outliers_upsample.py` — remove IQR outliers + upsample
- `truncate_downsample.py` — truncate to 500 words + downsample
- `truncate_upsample.py` — truncate to 500 words + upsample
- `pad_n_truncate_downsample.py` — pad short texts + truncate long + downsample
- `pad_n_truncate_upsample.py` — pad short texts + truncate long + upsample
- `split_downsample.py` — split long texts into chunks + downsample
- `split_upsample.py` — split long texts into chunks + upsample
All scripts use `MAX_LENGTH=500` by default.

**`Final Codes/tfidf-nn.ipynb`** — Ayush's final working notebook. **~80% macro-F1** on val. Uses TF-IDF features fed into a PyTorch MLP.

**`my_messy_kitchen/cook_here/codes/utils.py`** — The most sophisticated utility file in the project (465 lines). Contains:
- `stylometric_features()` — 15 features using `textstat` (Flesch reading ease, Gunning fog, syllable count) + hand-crafted (hedge words, bigram diversity, stopword ratio)
- `LLMClassifier` — DeBERTa encoder (CLS token) + stylometric features concatenated, fed to 2-layer MLP classifier (768 + 15 → 256 → 12)
- Full training loop with DataParallel, gradient clipping
- `predict_text()` — single-text inference with confidence scores
- `run_shap_explanation()` — SHAP KernelExplainer on CLS+stylo features (50 background, 20 test samples)
- `save_artifacts()` — saves label encoder + scaler

**`my_messy_kitchen/cook_here/codes/train_deberta.py`** — 36-run hyperparameter grid for DeBERTa:
- max_len: [256, 512, 128]
- lr: [2e-5, 3e-5, 1e-5]
- batch: [16, 32]
- lr_mode: [uniform, differential]
Uses Ayush's Kaggle dataset at `/kaggle/input/datasets/ayusheinsteinyadav/project`. Has SHAP. **NOT YET RUN.**

**`my_messy_kitchen/cook_here/codes/train_distilbert.py`** — DistilBERT fine-tuning.
**`my_messy_kitchen/cook_here/codes/train_roberta.py`** — RoBERTa fine-tuning.

**`my_messy_kitchen/cook_here/pipeline/pipeline1_best.py`** — MPNet + LinearSVC. Clean, production-quality. Loads data, cleans (strip whitespace, remove non-ASCII, dedup), encodes with `all-mpnet-base-v2`, trains LinearSVC(C=1), evaluates + saves confusion matrix. **NOT YET RUN on full data.**

**`my_messy_kitchen/cook_here/pipeline/pipeline2_sklearn.py`** — TF-IDF + 6 sklearn classifiers (LR, LinearSVC, ComplementNB, KNN, RF, GradientBoosting). Produces comparison bar chart + confusion matrix. **NOT YET RUN on full data.**

**`my_messy_kitchen/cook_here/pipeline/pipeline3_bpe_w2v.py`** — Custom tokenizers (BPE + WordPiece) × Word2Vec/FastText × LinearSVC. 4 combinations, trains tokenizers from scratch on training data. Uses gensim. Mean-pools word vectors. **NOT YET RUN on full data.**

---

### 2D. Aaditya's Work (structured notebooks + DistilBERT)

**`notebooks/` — 10-notebook structured series:**
- `01_eda_and_preprocessing.ipynb` — EDA
- `02_tfidf_classical_models.ipynb` — TF-IDF experiments
- `03_word_embeddings.ipynb` — Word2Vec, GloVe, FastText
- `04_sentence_embeddings.ipynb` — SBERT models comparison
- `05_deep_learning_rnn_cnn.ipynb` — LSTM, BiLSTM, GRU, TextCNN
- `06_transformers/` — progressive unfreezing (5 experiments: head-only → full finetune)
- `07_hybrid_features.ipynb` — SBERT + TF-IDF + Stylometric hybrid
- `08_ensemble.ipynb` — weighted ensemble
- `09_advanced_techniques.ipynb` — contrastive learning, perplexity features
- `10_results_analysis.ipynb` — final comparison
- `experiments.ipynb` — DistilBERT experiments

**`src/train.py`** — HF Trainer for DistilBERT. Takes model name as `sys.argv[1]`. 5 epochs, batch=8, fp16. Saves to `models/MODEL_NAME/`. ⚠️ Uses `fp16=torch.cuda.is_available()` — same fp16 risk as old DeBERTa.

**`src/evaluate.py`** — Evaluation pipeline.
**`src/predict.py`** — Single-text inference.
**`src/utils.py`** — Data loaders for the HF Trainer pipeline.

---

### 2E. `src/models/transformers/deberta.py` — The OLD (unfixed) DeBERTa module
248 lines. `TransformerTrainer` class with:
- 5 unfreezing strategies: `head_only`, `top2`, `top4`, `top6`, `full`
- `unfreeze_top_n_layers(model, n)` — unfreezes last N of 12 DeBERTa layers
- Differential LR: classifier head gets 10× LR vs encoder
- **Still uses `use_fp16 = self.device.type == 'cuda'` → THE BUG THAT CAUSED 8.29%**
- DataParallel + GradScaler.unscale_() incompatibility present
This is superseded by `deberta_finetune_FIXED.py`. Do NOT use this file for new runs.

---

### 2F. `src/evaluation/metrics.py` — Complete metrics module
- `compute_all_metrics()` — accuracy, F1 (macro+weighted), precision, recall, ROC-AUC, log_loss
- `plot_confusion_matrix()` — normalised heatmap
- `plot_calibration()` — per-class calibration curves
- `build_results_table()` — loads all JSON results files and makes comparison DataFrame
- `save_results()` — JSON serialization

---

### 2G. `llm_sbert_complete/` — The Massive SBERT Experiment Grid
**46 runs: 4 embeddings × (5 classical + 5 deep) classifiers**

Notebook: `llm_sbert_all_classifiers.ipynb`
Results: `Project_Output/Results/master_results.csv`
Saved models: `Project_Output/Saved_Models/` (40 `.joblib` files: BGE_BiLSTM.joblib etc.)
Embedding caches: `Project_Output/Cache/` (X_tr/vl/ts for MiniLM, MPNet, BGE, E5 as `.npy`)

**All 46 verified results (from master_results.csv):**

Classical heads (val accuracy):
| Embedding | LR | LinearSVC | RF | MLP | ComplementNB |
|---|---|---|---|---|---|
| MiniLM (384d) | 31.68% | 30.59% | 55.35% | 46.72% | 19.31% |
| MPNet (768d) | 35.41% | 35.75% | 56.42% | 52.13% | 15.77% |
| BGE (768d) | 41.01% | 41.45% | 56.98% | **58.43%** | 18.49% |
| E5 (768d) | 47.00% | 48.38% | 57.72% | **63.01%** | 23.15% |

Deep heads (val accuracy):
| Embedding | RNN | LSTM | BiLSTM | GRU | TextCNN |
|---|---|---|---|---|---|
| MiniLM | 31.17% | 39.58% | 41.08% | 38.97% | 31.07% |
| MPNet | 35.09% | 41.74% | 43.93% | 41.23% | 35.19% |
| BGE | 40.14% | 46.78% | 49.25% | 46.56% | 40.33% |
| E5 | 45.77% | 47.23% | **49.40%** | 47.12% | 46.30% |

**Key findings from this grid:**
1. **E5+MLP = 63.01%** — best in the grid. E5 encoder + MLP head dominates.
2. **BGE+MLP = 58.43%** — second best classical.
3. RandomForest overfits massively (99.99% train, 55–58% val).
4. MLP head always beats LR on embeddings (+15–20% absolute).
5. Deep RNN/LSTM models underperform MLP on this task (not sequential enough).
6. BiLSTM is the best deep architecture (49.25% BGE, 49.40% E5) but still below MLP.

---

### 2H. `llm-text-classifier-claudecode/llm-text-classifier/` — The Full Modular Scaffold

**`src/models/classical/all_classifiers.py`** — 9-classifier factory:
NB, LR, SVM (calibrated), RF, ExtraTrees, LightGBM, XGBoost, CatBoost, MLP (sklearn).
All configured from `config.yaml`. Supports GPU for LightGBM/XGBoost/CatBoost.

**`src/training/trainer.py`** — Universal `train_sklearn()` function: trains, evaluates on train/val/test, saves `.pkl` + result `.json`. Returns predictions + probabilities.

**`src/training/experiment_tracker.py`** — Tracks all experiment runs.

**`deployment/app.py`** — Gradio demo. Loads TF-IDF vectorizer + best model, predicts class + top-5 probabilities. Has 3 example texts (ChatGPT-style, informal human, repetitive GPT-2-style).

**`deployment/api.py`** — FastAPI REST endpoint. `/predict` (POST with `{"text": "..."}`) returns prediction + probabilities. `/health` (GET) returns status.

---

### 2I. `my_messy_kitchen/cook_here/pipeline/` — Ayush's 5 Production Pipelines

| Pipeline | Method | Status |
|---|---|---|
| `pipeline1_best.py` | MPNet + LinearSVC (clean end-to-end) | Ready, not run on full data |
| `pipeline2_sklearn.py` | TF-IDF + 6 sklearn classifiers (LR, SVC, NB, KNN, RF, GBM) | Ready, not run on full data |
| `pipeline3_bpe_w2v.py` | BPE/WordPiece × Word2Vec/FastText × LinearSVC (4 combos) | Ready, not run on full data |
| `pretrained.py` | Pre-trained transformer fine-tuning | In progress |
| `scratch.py` | Training from scratch | Experimental |

---

## SECTION 3 — COMPLETE VERIFIED RESULTS TABLE

| # | Model | Stage / Split | Val Acc | Source | Trust |
|---|---|---|---|---|---|
| 1 | TF-IDF + LinearSVM | **full (79,680 val)** | **78.71%** | results_all.json | ✅ BEST |
| 2 | TF-IDF + MLP (Ayush) | full val | **~80% F1** | tfidf-nn.ipynb | ✅ |
| 3 | E5 + MLP | medium split | 63.01% | master_results.csv | ✅ |
| 4 | BGE + MLP | medium split | 58.43% | master_results.csv | ✅ |
| 5 | MiniLM + LightGBM | 99,600 val | 58.06% | minilm_results.csv | ✅ |
| 6 | LightGBM + BGE + Stylo | mini | 60.88% | results_all.json | ✅ |
| 7 | E5 + BiLSTM | medium split | 49.40% | master_results.csv | ✅ |
| 8 | BGE + BiLSTM | medium split | 49.25% | master_results.csv | ✅ |
| 9 | TF-IDF + LinearSVM | mini (9,959 val) | 62.70% | results_all.json | ✅ |
| 10 | TF-IDF + LogReg | mini | 62.64% | results_all.json | ✅ |
| 11 | E5 + LinearSVC | medium split | 48.38% | master_results.csv | ✅ |
| 12 | MPNet + MLP | medium split | 52.13% | master_results.csv | ✅ |
| 13 | LR + BGE + Stylo | mini | 44.08% | results_all.json | ✅ |
| 14 | MiniLM + XGBoost | val | 43.19% | minilm_results.csv | ✅ |
| 15 | LR + MPNet + Stylo | mini | 41.07% | results_all.json | ✅ |
| 16 | TF-IDF + NaiveBayes | mini | 41.53% | results_all.json | ✅ |
| 17 | MiniLM + CatBoost | val | 38.30% | minilm_results.csv | ✅ |
| 18 | LR + MiniLM + Stylo | mini | 38.40% | results_all.json | ✅ |
| 19 | **DeBERTa-v3-base** | mini | **8.29%** | results_all.json | ❌ BROKEN |

### Numbers to NEVER cite:
- 100% in `mini_dataset_notebook2.ipynb` → data leakage
- 99.6% LightGBM in MiniLM grid → train accuracy (val=58%)
- 95–96% in root README.md → template, not measured
- Any 91–97% claim in README.txt → tutorial ranges, never achieved here

---

## SECTION 4 — THE DEBERTA STORY (Complete Root-Cause)

### Bug Chain:
1. **fp16 overflow** on Turing T4 → first gradient is `inf`/`NaN`
2. **Guard only checked loss** (`isnan(loss)`) — never gradients
3. `clip_grad_norm_` on `inf` = no-op → `optimizer.step()` → NaN in all weights
4. Weights NaN → every forward NaN → loss skipped → `train_loss` decays to 0
5. Model predicts same class forever → 8.29% (= 1/12 = random)

### All 5 Bugs Fixed in `deberta_finetune_FIXED.py`:
| Bug | Problem | Fix |
|---|---|---|
| A | gradient_checkpointing + DataParallel crashes | disabled when DataParallel |
| B | LR 2e-5 encoder / 20e-5 head = too high | 1e-5 encoder / 5e-5 head (3×) |
| C | Gradient check on loss only | `_finite_grads()` checks all grad tensors |
| D | max_length=512 on 14.6GB T4 = OOM | max_length=256 |
| E | GradScaler.unscale_() + DataParallel = ValueError | fp32 forced when multi-GPU |

### Same risk in `src/train.py` (Aaditya):
Line 49: `fp16=torch.cuda.is_available()` → will fail on T4 for DeBERTa-v3. Safe for DistilBERT (more stable in fp16) but monitor carefully.

### Same risk in `src/models/transformers/deberta.py` (old module):
Line 150: `use_fp16 = self.device.type == 'cuda'` → DO NOT USE THIS FILE for DeBERTa-v3 training.

---

## SECTION 5 — INDUSTRY-READY PROJECT STRUCTURE

### Target Folder Structure (after cleanup):
```
llm-authorship-attribution/         ← repo root
│
├── README.md                        ← REWRITE with real numbers
├── requirements.txt                 ← already good
├── setup.py                         ← already exists
├── Makefile                         ← NEW: one-command shortcuts
├── .gitignore                       ← update (block parquet/pkl/pt/npy/zip)
│
├── config/
│   └── config.yaml                  ← unified config for all pipelines
│
├── data/
│   ├── README.md                    ← how to download RAID dataset
│   ├── demo/                        ← Dataset_small + Dataset_medium (commit these)
│   │   ├── small/{train,val,test}/
│   │   └── medium/{train,val,test}/
│   └── full/                        ← GITIGNORED — user puts LLM_12_class/raid/ here
│       ├── train/train.parquet
│       ├── val/val.parquet
│       └── test/test.parquet
│
├── src/
│   ├── data/
│   │   ├── loader.py                ← unified load (handles both column schemes)
│   │   └── preprocess.py            ← clean_text, truncate, normalise
│   ├── features/
│   │   ├── tfidf.py
│   │   ├── stylometric.py           ← 40-feature extractor (keep Arjun's version)
│   │   └── sentence_embeddings.py   ← SBERT/BGE/E5 with disk caching
│   ├── models/
│   │   ├── classical.py             ← all 9 classifiers (from all_classifiers.py)
│   │   ├── deep.py                  ← DNN, CNN, BiLSTM (Affan's architectures)
│   │   └── deberta.py               ← THE FIXED VERSION ONLY
│   ├── training/
│   │   └── trainer.py               ← universal train/eval/save loop
│   └── evaluation/
│       ├── metrics.py               ← already good (src/evaluation/metrics.py)
│       └── plots.py                 ← confusion matrix, calibration, comparison
│
├── scripts/
│   ├── 01_smoke_test.py             ← full pipeline on demo data, <60s
│   ├── 02_train_classical.py        ← TF-IDF + SVM/LR/NB on FULL data
│   ├── 03_train_embeddings.py       ← E5/BGE + MLP/LGBM on FULL data
│   ├── 04_train_deberta.py          ← deberta_finetune_FIXED (integrated)
│   ├── 05_train_ensemble.py         ← weighted ensemble
│   └── 06_final_evaluation.py       ← TEST SET ONCE — final numbers
│
├── notebooks/
│   ├── 01_eda.ipynb                 ← EDA with real outputs
│   ├── 02_classical_baselines.ipynb ← TF-IDF experiments
│   ├── 03_embeddings.ipynb          ← embedding comparison (46-run grid)
│   ├── 04_deberta_finetune.ipynb    ← DeBERTa training + the fix story
│   └── 05_results_comparison.ipynb  ← final comparison table + plots
│
├── deployment/
│   ├── app.py                       ← Gradio demo (already written, fix model path)
│   └── api.py                       ← FastAPI REST endpoint (already written)
│
├── results/
│   ├── results_all.json             ← COMMIT — verified results
│   ├── master_results.csv           ← COMMIT — 46-run SBERT grid
│   └── plots/                       ← COMMIT — all PNG plots
│
└── tests/
    └── test_smoke.py                ← import + predict tests
```

---

## SECTION 6 — WHAT TO COMMIT vs NOT

| File/Folder | Commit? | Reason |
|---|---|---|
| `src/` all Python | ✅ | core code |
| `scripts/01-06_*.py` | ✅ | reproducibility |
| `notebooks/01-05_*.ipynb` (with outputs) | ✅ | shows work |
| `config/config.yaml` | ✅ | configuration |
| `results/results_all.json` | ✅ | verified results |
| `results/master_results.csv` | ✅ | SBERT grid |
| `results/plots/*.png` | ✅ | visuals |
| `data/demo/` (small parquets) | ✅ | enables smoke test |
| `deployment/app.py` + `api.py` | ✅ | shows deployment skill |
| `tests/test_smoke.py` | ✅ | engineering credibility |
| `requirements.txt`, `setup.py` | ✅ | setup |
| `data/full/*.parquet` | ❌ | 800MB+, gitignored |
| `models/*.pkl`, `*.pt`, `*.joblib` | ❌ | 100MB+, gitignored |
| `*.npy` embedding caches | ❌ | 500MB+, gitignored |
| `_archive/` (duplicates) | ❌ | gitignored |
| `hf_cache/`, `.conda/` | ❌ | gitignored |
| `results.zip`, `Dataset.zip` | ❌ | gitignored |
| All duplicate notebooks | ❌ | archive first |

---

## SECTION 7 — THE COMPLETE PRIORITY PLAN

### Step 1 — Repo Cleanup (2–3 hours, local, no GPU needed)
Create `_archive/` and move all duplicate notebooks into it:
```
llm-minilm-classifier*.ipynb   llm_classifier_*.ipynb   llm-classifier-*.ipynb
llm-text-classifier-bge.ipynb  llm_text_classifier_*.ipynb
llm-local-pc-complete.ipynb    llm_local_complete/
llm_memory_safe*/              memory_safe_embeddings_kaggle.py
mini_dataset_notebook*.ipynb   notebook1.ipynb   train_dataset_notebook1.ipynb
LLM_Classifier_Complete.ipynb  LLM_Text_Classifier_Complete.ipynb
LLM_Classifier_E5_Full_Kaggle.ipynb  llm_classifier_full_optimizedANKIT.ipynb
llm_mpnet_classifier_final.ipynb  llm_minilm_fixed_cell5.ipynb
EDA_mini_train*.ipynb   demo.ipynb   download*.txt   manage_github_repo_working.txt
```

### Step 2 — Fix the README NOW (30 minutes)
Delete the fake numbers (96.2%, 95.8%, etc.). Replace with the real table from Section 3.
The README must be honest before any push to GitHub.

### Step 3 — Write the Unified Loader (1 hour)
`src/data/loader.py` — single function that normalises `(model, generation)` vs `(generated_by, text)`:
```python
def load_split(path, subset=None):
    df = pd.read_parquet(path)
    if 'generated_by' in df.columns:
        df = df.rename(columns={'generated_by':'model','text':'generation'})
    df = df[df['model'].isin(CLASSES)].reset_index(drop=True)
    if subset:  # 'mini' = 1000/class
        df = df.groupby('model').head(subset).reset_index(drop=True)
    return df
```

### Step 4 — Write Smoke Test (1 hour)
`scripts/01_smoke_test.py` — runs on `data/demo/small/` (no GPU, <60s):
- Load data → check 12 classes present
- TF-IDF + SVM → check val_acc > 50%
- Stylometric extractor → check shape (N, 40)
- PASS/FAIL printout

### Step 5 — Run DeBERTa Fix on GPU (2 hours, Kaggle P100 or Colab A100)
Upload `deberta_finetune_FIXED.py` + the RAID dataset to Kaggle.
```python
from deberta_finetune_FIXED import run
model, tok, le, acc = run(X_train, y_train, X_val, y_val, classes=CLASSES)
# Expected: mini ~60-75%, full ~85-92%
```
Record result in `results/results_all.json` immediately.

### Step 6 — Run Pipelines 1/2/3 on Full Data (1 day, CPU or GPU)
These 3 scripts are ready and clean:
- `my_messy_kitchen/cook_here/pipeline/pipeline1_best.py` → MPNet+LinearSVC on full RAID
- `my_messy_kitchen/cook_here/pipeline/pipeline2_sklearn.py` → TF-IDF+6 classifiers on full RAID
- `my_messy_kitchen/cook_here/pipeline/pipeline3_bpe_w2v.py` → BPE/WordPiece×W2V/FastText on full RAID
Add results to `results/results_all.json`.

### Step 7 — Touch the Test Set ONCE (½ hour)
`scripts/06_final_evaluation.py` — load every saved model, run on test set, record:
| Model | Val Acc | Test Acc | F1 | Train Time |
|---|---|---|---|---|
| TF-IDF + SVM | 78.71% | **?** | ? | 25s CPU |
| TF-IDF + MLP | ~80% | **?** | ? | mins CPU |
| E5 + MLP | 63.01% | **?** | ? | GPU embed |
| DeBERTa (fixed) | ? | **?** | ? | ~2h GPU |
| Ensemble | ? | **?** | ? | sum |

### Step 8 — Deploy Gradio to HuggingFace Spaces (1 hour)
Fix `deployment/app.py` to load the saved TF-IDF+SVM model. Deploy:
```bash
huggingface-cli login
huggingface-cli repo create llm-authorship-demo --type space --space_sdk gradio
```
This gives you a permanent public URL for your resume: `huggingface.co/spaces/[username]/llm-authorship-demo`

### Step 9 — Update README with Final Numbers + Push to GitHub
Fill in all `XX.XX%` placeholders with real test results. Push.

---

## SECTION 8 — CRITICAL GOTCHAS (Every One Found by Reading Code)

| # | Gotcha | Where | Impact |
|---|---|---|---|
| 0 | ~~val.parquet == test.parquet~~ **✅ FIXED** — split 50/50 stratified | `LLM_12_class/raid/` | New: val=49,800, test=49,800, no overlap |
| 1 | **fp16 for DeBERTa-v3 = NaN** | `src/models/transformers/deberta.py:150` | 8.29% accuracy |
| 2 | **Column names differ**: `(model,generation)` vs `(generated_by,text)` | Dataset_small/medium vs RAID | Crash on load |
| 3 | **GradScaler.unscale_() + DataParallel = ValueError** | old deberta.py | Crash on dual GPU |
| 4 | **Unicode chars in data**: `​` zero-width space | full RAID dataset | UnicodeEncodeError |
| 5 | **RandomForest overfits completely** on embeddings | master_results.csv | 99.99% train, 55% val |
| 6 | **MiniLM LightGBM train=99.6% but val=58%** | minilm_results.csv | Don't report train acc |
| 7 | **embed.py reads from `../Dataset_cleaned`** (Affan) | embed.py:14 | File not found on your machine |
| 8 | **train.py reads from `./embeddings/`** (Affan) | train.py:7 | Need to run embed.py first |
| 9 | **pipeline scripts read `train.parquet` from `.`** (Ayush) | pipeline1/2/3.py | Need to copy RAID files |
| 10 | **train_deberta.py data path is Kaggle-specific** | train_deberta.py:58 | Need to update path |
| 11 | **Full train set is 796,800 not 960,000** | verified from parquet | Report honestly |
| 12 | **test set must only be touched once** | project brief | Don't iterate on test |
| 13 | **E5 requires "passage:" prefix** | embedding_models.py:62 | Wrong embeddings without prefix |
| 14 | **Qwen3 requires "query" prompt** | embedding_models.py:22 | Wrong for document encoding |

---

## SECTION 9 — RESUME LINE (fill in test accuracy after Step 7)

```
LLM Authorship Attribution — 12-Class Text Classification            IIT Dharwad, CS204T
• Built a pipeline to identify which of 12 LLMs (GPT-2/3/4, LLaMA, Mistral, etc.) or a
  human authored a given text — 796,800-row RAID dataset, 12 perfectly-balanced classes.
• Best result: XX.XX% test accuracy with DeBERTa-v3 fine-tune; TF-IDF+SVM baseline = 78.71%
  (CPU-only, 25s training time).
• Diagnosed and fixed a fp16 NaN weight-poisoning bug that caused total model collapse
  (8.29% = random); implemented finite-gradient guards, bf16/fp32 auto-selection, and
  sanity-overfit checks.
• Compared 6 model families (TF-IDF, word-embeddings, sentence-embeddings, DNN/CNN, BiLSTM,
  DeBERTa) across accuracy, training time, and computational cost.
• Stack: PyTorch, HuggingFace Transformers, scikit-learn, LightGBM, Gradio, FastAPI.
• GitHub: [link] | Live demo: [HuggingFace Spaces link]
```

---

## SECTION 10 — QUICK REFERENCE

| What | Exact Path |
|---|---|
| **Real full dataset** | `LLM_12_class/raid/{train,val,test}/*.parquet` |
| **Demo data** | `Dataset/Dataset_small/` + `Dataset/Dataset_medium/` |
| **DeBERTa fix (ready to run)** | `deberta_finetune_FIXED.py` |
| **Canonical pipeline notebook** | `llm-generated-text-2.ipynb` |
| **Verified results ledger** | `TF-IDF/results/models/results_all.json` |
| **46-run SBERT grid results** | `llm_sbert_complete/Project_Output/Results/master_results.csv` |
| **Best saved model** | `TF-IDF/results/models/full_tfidf_svm.pkl` |
| **Stylometric features (40)** | `src/features/stylometric.py` |
| **Ayush comprehensive utils** | `my_messy_kitchen/cook_here/codes/utils.py` |
| **3 production pipelines** | `my_messy_kitchen/cook_here/pipeline/pipeline{1,2,3}_*.py` |
| **DeBERTa 36-run grid** | `my_messy_kitchen/cook_here/codes/train_deberta.py` |
| **Ayush TF-IDF+MLP** | `Final Codes/tfidf-nn.ipynb` |
| **Affan embedding generator** | `embed.py` |
| **Affan DNN+CNN** | `models.py` + `train.py` |
| **Gradio demo** | `llm-text-classifier-claudecode/llm-text-classifier/deployment/app.py` |
| **FastAPI endpoint** | `llm-text-classifier-claudecode/llm-text-classifier/deployment/api.py` |
| **All-classifiers factory** | `llm-text-classifier-claudecode/llm-text-classifier/src/models/classical/all_classifiers.py` |
| **Old (broken) DeBERTa** | `src/models/transformers/deberta.py` ← DO NOT USE |
