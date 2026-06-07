import numpy as np
import os
from pathlib import Path

CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODELS = {
    "MiniLM":    "all-MiniLM-L6-v2",
    "MPNet":     "all-mpnet-base-v2",
    "BGE-base":  "BAAI/bge-base-en-v1.5",
    "BGE-large": "BAAI/bge-large-en-v1.5",
    "E5-base":   "intfloat/e5-base-v2",
    "E5-large":  "intfloat/e5-large-v2",
}


def get_embeddings(texts, model_name, cache_key, batch_size=256,
                   normalize=True, max_chars=2048):
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        emb = np.load(cache_path)
        print(f"  Loaded from cache: {cache_path.name} → {emb.shape}")
        return emb

    from sentence_transformers import SentenceTransformer
    import torch

    print(f"  Encoding {len(texts):,} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    truncated = [t[:max_chars] for t in texts]

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        pool = model.start_multi_process_pool(
            target_devices=[f'cuda:{i}' for i in range(n_gpus)]
        )
        emb = model.encode_multi_process(
            truncated, pool=pool, batch_size=batch_size,
            normalize_embeddings=normalize
        )
        model.stop_multi_process_pool(pool)
    else:
        emb = model.encode(
            truncated, batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=normalize,
            convert_to_numpy=True
        )

    emb = np.array(emb, dtype=np.float32)
    np.save(cache_path, emb)
    print(f"  Shape: {emb.shape} — saved to {cache_path.name}")
    return emb