import numpy as np
from pathlib import Path

GLOVE_PATH = Path("data/external/glove.840B.300d.txt")
FASTTEXT_PATH = Path("data/external/crawl-300d-2M-subword.bin")


def load_glove(path=GLOVE_PATH):
    print(f"Loading GloVe from {path}...")
    vecs = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            try:
                vecs[word] = np.array(parts[1:], dtype=np.float32)
            except:
                pass
    print(f"  Loaded {len(vecs):,} word vectors")
    return vecs


def text_to_mean_vector(text, word_vecs, dim=300):
    tokens = text.lower().split()
    vecs = [word_vecs[t] for t in tokens if t in word_vecs]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 0 else v


def texts_to_mean_vectors(texts, word_vecs, dim=300):
    return np.vstack([text_to_mean_vector(t, word_vecs, dim) for t in texts])


def get_fasttext_embeddings(texts, model_path=FASTTEXT_PATH, dim=300):
    try:
        import fasttext
        model = fasttext.load_model(str(model_path))
        embs = np.vstack([
            model.get_sentence_vector(t[:500].replace('\n', ' '))
            for t in texts
        ])
        return embs.astype(np.float32)
    except ImportError:
        print("Install fasttext: pip install fasttext-wheel")
        return np.random.randn(len(texts), dim).astype(np.float32)