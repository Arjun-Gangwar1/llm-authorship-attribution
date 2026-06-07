# How to Run the Full Pipeline on Kaggle

## What this runs
- Stage 1: TF-IDF + LinearSVM (best classical — prev. 78.71%)
- Stage 2: E5 embeddings + MLP (best embedding — prev. 63%)
- Stage 3: DeBERTa-v3-base fine-tune (ALL 5 BUGS FIXED — was broken at 8.29%)
- Stage 4: Weighted Ensemble (DeBERTa 65% + E5-MLP 25% + SVM 10%)
- Stage 5: Final results table + plots

**Estimated time: ~3–4 hours on T4×2 GPU**

---

## Step-by-Step

### Step 1 — Upload your dataset to Kaggle

Option A — Use the existing RAID dataset (easiest):
1. Go to kaggle.com → Datasets → Search `liamdugan raid`
2. Note the dataset path (it will be `/kaggle/input/raid/`)

Option B — Upload your local dataset:
1. Kaggle → Datasets → New Dataset
2. Upload the 3 parquet files from `LLM_12_class/raid/`:
   - `train/train.parquet`
   - `val/val.parquet`
   - `test/test.parquet`
3. Name it `llm-authorship-raid`
4. The path will be `/kaggle/input/llm-authorship-raid/`

### Step 2 — Create a new Kaggle Notebook

1. Go to kaggle.com → Notebooks → New Notebook
2. Click **File → Import Notebook**
3. Upload `kaggle_full_pipeline.ipynb` from this project

### Step 3 — Attach your dataset

In the notebook sidebar:
1. Click **+ Add Data**
2. Search for your dataset (from Step 1)
3. Add it

### Step 4 — Update the data path (if needed)

In **Cell 2 (CONFIG)**, update `DATA_DIR` to match your dataset path:

```python
# If you used Option A (liamdugan/raid):
DATA_DIR = '/kaggle/input/raid'

# If you uploaded your own (Option B):
DATA_DIR = '/kaggle/input/llm-authorship-raid'
```

### Step 5 — Set GPU accelerator

1. Right sidebar → **Settings**
2. **Accelerator** → Select **GPU T4 x2**
3. Click **Save**

### Step 6 — Run All

Click **Run All** (▶▶) and wait ~3–4 hours.

You'll see live output like:
```
STAGE 1: TF-IDF BASELINES
  TF-IDF + LinearSVM: val=XX.XX%  test=XX.XX%  ✅

STAGE 2: E5 EMBEDDINGS + MLP
  Epoch 1/15 | val_acc=52.xx%
  ...
  E5 + MLP: val=XX.XX%  test=XX.XX%  ✅

STAGE 3: DeBERTa-v3-base FINE-TUNE (FIXED)
  [sanity] 16-sample overfit = 100.0% (OK)
  Epoch 1/3 | train_loss=1.xxxx | val_loss=0.xxxx | val_acc=62.xx%
  Epoch 2/3 | train_loss=0.xxxx | val_loss=0.xxxx | val_acc=79.xx%
  Epoch 3/3 | train_loss=0.xxxx | val_loss=0.xxxx | val_acc=87.xx%  ← target

STAGE 4: WEIGHTED ENSEMBLE
  Ensemble: val=XX.XX%  test=XX.XX%  ✅

STAGE 5: FINAL RESULTS TABLE ✅
```

### Step 7 — Download Outputs

1. Kaggle → Your Notebook → **Output tab**
2. Download All (or individual files)

Files you'll get:
```
results_all.json              ← all measured accuracies
final_comparison_table.csv    ← comparison table
tfidf_vectorizer.pkl          ← saved TF-IDF
tfidf_svm.pkl                 ← saved SVM (best classical)
e5_mlp_best.pt                ← saved E5+MLP
deberta_best.pt               ← saved DeBERTa (best model)
confusion_matrix_ensemble.png ← confusion matrix plot
deberta_training_curves.png   ← DeBERTa loss/accuracy curves
model_comparison.png          ← comparison bar chart + cost scatter
```

### Step 8 — Save results back to the project

Copy `results_all.json` and `final_comparison_table.csv` to:
```
TF-IDF/results/models/results_all.json   ← overwrite with new results
results/final_comparison_table.csv
results/plots/confusion_matrix_ensemble.png
results/plots/deberta_training_curves.png
results/plots/model_comparison.png
```

Then update README.md with the real test numbers.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `FileNotFoundError: train.parquet` | Update `DATA_DIR` in Cell 2 to match your dataset path |
| `CUDA out of memory` | Reduce `batch_size` from 16 to 8 in `DEBERTA_CONFIG` |
| `[sanity] FAIL` | Check dataset loaded correctly — print `X_train[:2]` in Cell 3 |
| `val_loss = nan` after epoch 1 | Still an fp16 issue — make sure `USE_AMP=False` (should auto-detect) |
| DeBERTa still at 8.29% after epoch 1 | Check `skipped_steps` count — if high, reduce lr to 5e-6 |
| Session crashes (OOM) | Enable `gradient_checkpointing` and reduce `max_length` to 128 |

---

## Expected Results

| Model | Val Acc | Test Acc |
|---|---|---|
| TF-IDF + LinearSVM | ~79–82% | ~78–81% |
| E5 + MLP | ~68–75% | ~67–74% |
| DeBERTa-v3-base | ~85–92% | ~84–91% |
| Ensemble | ~87–93% | ~86–92% |
