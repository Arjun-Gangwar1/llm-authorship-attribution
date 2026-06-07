# LLM Authorship Attribution — consolidated notes (single source of truth)

> Written after a full sweep of Desktop + Downloads (40+ notebook copies). Purpose:
> stop you getting lost among duplicates, record what's *actually verified*, and
> give the exact steps to (a) fix DeBERTa and (b) update the portfolio afterward.
> Last updated 2026-06-02.

---

## 1. The task (authoritative — from the CS204T brief PDF)
- 12-class authorship attribution, dataset derived from Hugging Face **RAID**, **perfectly balanced**.
- Splits: **train 960,000 / val 99,600 / test 99,600**. Columns: `model` (label), `generation` (text).
- Classes: `gpt2, gpt3, gpt4, chatgpt, llama-chat, mistral, mistral-chat, mpt, mpt-chat, cohere, cohere-chat, human`.
- Mini subset = 1,000 rows/class for fast debugging.

## 2. VERIFIED results (measured, in saved notebook outputs)
| Model | Split | Accuracy | Notes |
|---|---|---|---|
| **TF-IDF + Linear SVM** | full (79,680 val) | **78.71%** | ✅ best legitimate result; per-class F1: gpt4 0.89, chatgpt 0.88, llama-chat 0.86, human 0.82, gpt3 0.81 |
| TF-IDF + MLP (NN) | 99,600 | ~80% macro-F1 | ✅ teammate (Ayush) run |
| TF-IDF + LogReg | mini 9,959 | 62.64% | |
| LightGBM + BGE-base + stylo | mini | 60.88% | best embedding combo |
| SBERT (MiniLM) + LightGBM/BiLSTM | 99,587 | 52–58% | val acc (train was 99% = overfit) |
| LR + BGE/MPNet/MiniLM + stylo | mini | 38–44% | |
| **DeBERTa-v3 + ensemble** | any | **8.29%** | ❌ BROKEN (random = 1/12). See §4. |

**Best honest headline = 78.7%.** Embeddings converged on **BGE-base** (not E5 — E5 ≤68% in every run).

## 3. ⚠️ Numbers you must NOT trust
- **"100%" in `mini_dataset_notebook2.ipynb`** — runs on a tiny 9,600/2,400 split where *every* model scores 100% → **data leakage**. Not real.
- **"99.6%" in `llm-minilm-classifier (3).ipynb`** — that's LightGBM **train** accuracy (overfit); real val = 58%.
- **"95–96%" in `llm-text-classifier/README.md`** — aspirational template; that repo's notebooks are empty (0 bytes). No executed backing.

## 4. DeBERTa root cause + the fix
**Symptom:** `train_loss → 0.0000`, `val_loss = nan`, `val_acc = 8.29%` (constant-class prediction).
**Cause chain:** fp16 on Turing T4 → an early gradient goes `inf` → the old guard only checked `loss` (not grads) and `clip_grad_norm_` is a no-op on an inf norm → `opt.step()` wrote **NaN into all weights** → permanent collapse; every later batch is NaN and skipped, so `train_loss` decays to 0.
**Fix:** see **`deberta_finetune_FIXED.py`** (same folder). Key changes: bf16-or-fp32 (never fp16); a real finite-gradient guard that drops the step instead of poisoning weights; clip every step; conservative LR; one shared LabelEncoder with range asserts; a `sanity_overfit()` fail-fast on 16 samples.

**Run it (Kaggle/Colab with GPU):**
```python
from deberta_finetune_FIXED import run
model, tok, le, acc = run(X_mini_train, y_mini_train, X_mini_val, y_mini_val, classes=CLASSES)
```
Expect: `Precision: bf16|fp32` (never fp16) → `[sanity] ... = 100.0% (OK)` → `val_acc` climbing past 8.3% (mini ~60–75%, full ~85–92%), `skipped_steps` low, `val_loss` a real number.

## 5. Which files are canonical (ignore the rest)
- **Code of record:** `llm-classifier/Arjun/llm-generated-text-2.ipynb` (full executed pipeline + outputs) and `results_all.json`.
- **Stylometric features (40):** `llm-text-classifier/src/features/stylometric.py`.
- **DeBERTa fix:** `ml lab project/deberta_finetune_FIXED.py` (this folder).
- **Team repo:** https://github.com/ItzCobaltboy/llm-classifier
- Everything else in `Downloads/llm*` / `llm-text-gen_downloaded_files/` are older iterations/duplicates — safe to archive.

## 6. After DeBERTa works → update the portfolio
File: `Desktop/personal_portfolio/project-llm-attribution.html`
1. In the **Full stage** results table, add a `DeBERTa-v3 (fine-tuned)` row and a `Weighted ensemble (0.65/0.25/0.10)` row with the real accuracy/F1.
2. Replace the **"An honest note on DeBERTa"** blockquote with a short "what was wrong and how I fixed it" paragraph (the NaN-from-fp16 story is itself a great interview talking point).
3. Update the home/projects metric chips if the headline accuracy changes.
4. If E5 still loses to BGE-base, change the résumé line "converging on E5" → "converging on BGE-base (E5 also evaluated)".
