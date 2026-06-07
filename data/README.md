# Data Directory

## Structure

```
data/
├── demo/                  ← Committed. Use for smoke tests (no download needed).
│   ├── small/             ← ~664 rows/class, columns: text, generated_by
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── medium/            ← ~6,640 rows/class, columns: text, generated_by
│       ├── train/
│       ├── val/
│       └── test/
└── full/ → LLM_12_class/raid/   ← NOT committed (gitignored). Download below.
    ├── train/train.parquet       (796,800 rows)
    ├── val/val.parquet           (49,800 rows)
    └── test/test.parquet         (49,800 rows)
```

## Download the Full Dataset

The full dataset is the RAID benchmark from HuggingFace:

```python
from datasets import load_dataset
import os

os.makedirs("LLM_12_class/raid/train", exist_ok=True)
os.makedirs("LLM_12_class/raid/val",   exist_ok=True)
os.makedirs("LLM_12_class/raid/test",  exist_ok=True)

ds = load_dataset("liamdugan/raid")
ds["train"].to_parquet("LLM_12_class/raid/train/train.parquet")
ds["validation"].to_parquet("LLM_12_class/raid/val/val.parquet")
ds["test"].to_parquet("LLM_12_class/raid/test/test.parquet")
```

## Column Names

| Dataset | Text column | Label column |
|---|---|---|
| `data/demo/` | `text` | `generated_by` |
| `LLM_12_class/raid/` | `generation` | `model` |

The unified loader (`src/data/loader.py`) handles both automatically.

## Quick Test (no download needed)

```bash
python scripts/01_smoke_test.py
```
