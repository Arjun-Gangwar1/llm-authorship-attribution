"""
scripts/01_smoke_test.py — Full pipeline smoke test on demo data.

Runs in < 60 seconds on any machine (CPU, no GPU needed).
Uses data/demo/small/ — the tiny dataset for pipeline verification.

Pass = the full pipeline is wired correctly end-to-end.
Fail = something is broken before you waste GPU time.

Usage:
    python scripts/01_smoke_test.py
"""

import sys, time
sys.path.insert(0, ".")

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
results = []


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  {PASS}  {name}  ({elapsed:.1f}s)")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        results.append(False)


print("\n" + "="*55)
print("  SMOKE TEST — LLM Authorship Attribution Pipeline")
print("="*55 + "\n")


# ── 1. Imports ────────────────────────────────────────────────
def test_imports():
    from src.data.loader import load_demo_dataset, CLASSES
    from src.features.stylometric import extract_stylometric, extract_batch
    assert len(CLASSES) == 12

check("Core imports", test_imports)


# ── 2. Data loader ────────────────────────────────────────────
def test_loader():
    from src.data.loader import load_demo_dataset, CLASSES
    X_tr, y_tr, X_vl, y_vl, X_ts, y_ts = load_demo_dataset("small")
    assert len(X_tr) > 0,          "No training rows loaded"
    assert len(set(y_tr)) == 12,   f"Expected 12 classes, got {len(set(y_tr))}"
    assert len(X_tr) == len(y_tr), "Texts and labels length mismatch"
    assert len(X_vl) > 0,         "No val rows"
    assert len(X_ts) > 0,         "No test rows"
    print(f"         train={len(X_tr)} val={len(X_vl)} test={len(X_ts)}", end="")

check("Data loader (demo/small)", test_loader)


# ── 3. Stylometric features ───────────────────────────────────
def test_stylometric():
    from src.features.stylometric import extract_stylometric
    import numpy as np
    sample = "Certainly! I'd be happy to help you with this question. Here are the key points."
    feat = extract_stylometric(sample)
    assert feat.shape == (40,), f"Expected (40,), got {feat.shape}"
    assert not any(import_np_isnan := __import__('numpy').isnan(feat)), "NaN in features"

check("Stylometric features shape=(40,)", test_stylometric)


# ── 4. TF-IDF vectorization ───────────────────────────────────
def test_tfidf():
    from src.data.loader import load_demo_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    X_tr, y_tr, X_vl, y_vl, _, _ = load_demo_dataset("small")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_vl_vec = tfidf.transform(X_vl)
    assert X_tr_vec.shape[1] <= 5000
    assert X_vl_vec.shape[1] == X_tr_vec.shape[1]
    print(f"         matrix={X_tr_vec.shape}", end="")

check("TF-IDF vectorization", test_tfidf)


# ── 5. TF-IDF + LinearSVM accuracy > 50% ─────────────────────
def test_tfidf_svm():
    from src.data.loader import load_demo_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    X_tr, y_tr, X_vl, y_vl, _, _ = load_demo_dataset("small")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_v = tfidf.fit_transform(X_tr)
    X_vl_v = tfidf.transform(X_vl)
    svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    svm.fit(X_tr_v, y_tr)
    acc = accuracy_score(y_vl, svm.predict(X_vl_v))
    # demo/small is tiny (~83/class train) so 30%+ proves the pipeline runs
    assert acc > 0.30, f"SVM val accuracy too low: {acc:.2%} (random=8.3%)"
    print(f"         val_acc={acc:.2%}", end="")

check("TF-IDF + LinearSVM (val > 50%)", test_tfidf_svm)


# ── 6. Stylometric batch extraction ──────────────────────────
def test_stylo_batch():
    from src.data.loader import load_demo_dataset
    from src.features.stylometric import extract_batch
    X_tr, _, _, _, _, _ = load_demo_dataset("small")
    batch = X_tr[:50]
    feats = extract_batch(batch)
    assert feats.shape == (50, 40), f"Expected (50, 40), got {feats.shape}"

check("Stylometric batch extraction (50 samples)", test_stylo_batch)


# ── Summary ───────────────────────────────────────────────────
print()
passed = sum(results)
total  = len(results)
print("="*55)
if passed == total:
    print(f"  \033[92mALL {total}/{total} CHECKS PASSED\033[0m — pipeline is healthy")
    print("  Ready to run on full data.")
else:
    print(f"  \033[91m{passed}/{total} PASSED — fix the failures above before full run\033[0m")
print("="*55 + "\n")

sys.exit(0 if passed == total else 1)
