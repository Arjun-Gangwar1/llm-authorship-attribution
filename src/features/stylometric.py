import re
import numpy as np
from collections import Counter

LLM_PATTERNS = {
    'certainly':     r'\bcertainly\b|\bof course\b',
    'as_an_ai':      r'\bas an ai\b|\blanguage model\b',
    'great_q':       r'\bgreat question\b|\bexcellent question\b',
    'happy_help':    r"happy to help|i'?d be happy",
    'however':       r'\bhowever\b',
    'furthermore':   r'\bfurthermore\b|\bmoreover\b|\badditionally\b',
    'informal':      r'\btbh\b|\bidk\b|\bimo\b|\blol\b|\bhaha\b',
    'you_know':      r'\byou know\b|\bi mean\b|\bkind of\b',
    'contractions':  r"\b\w+n't\b|\b\w+'re\b|\b\w+'ve\b",
    'repetition_3x': r'(\b\w{3,}\b)\s+\1\s+\1',
    'primary_core':  r'\bprimary\b|\bfundamental\b|\bcore\b',
    'methodology':   r'\bmethodology\b|\bframework\b|\bworkflow\b',
    'thank_you':     r'\bthank you\b|\bi believe\b',
    'let_me':        r'\blet me\b|\blet me explain\b',
    'here_are':      r'\bhere are\b|\bhere is\b',
}

def extract_stylometric(text: str) -> np.ndarray:
    if not isinstance(text, str) or len(text.strip()) < 5:
        return np.zeros(40, dtype=np.float32)

    tl = text.lower()
    words = text.split()
    alpha_words = [w for w in words if w.isalpha()]
    nw = max(len(words), 1)
    sents = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    ns = max(len(sents), 1)

    wl  = [len(w) for w in alpha_words]
    sl  = [len(s.split()) for s in sents]
    uq  = len(set(w.lower() for w in alpha_words))

    feats = [
        len(text),
        nw, ns,
        np.mean(wl) if wl else 0,
        uq / max(nw, 1),                          # vocab richness (TTR)
        np.mean(sl) if sl else 0,                  # avg sent len
        np.std(sl)  if len(sl) > 1 else 0,        # std sent len
        max(sl) if sl else 0,                      # max sent len
        sum(1 for c in text if c.isupper()) / nw, # upper char ratio
        text.count(',')  / nw,
        text.count('!')  / ns,
        text.count('?')  / ns,
        text.count(':')  / ns,
        text.count(';')  / ns,
        text.count('...') / ns,
        text.count('(')  / ns,
        text.count('"')  / nw,
        len(re.findall(r'^\s*[-•*]\s', text, re.M)) / ns,
        len(re.findall(r'^\s*\d+[.)]\s', text, re.M)) / ns,
        text.count('\n') / ns,
        sum(1 for w in alpha_words if len(w) > 8) / max(nw,1),  # long words ratio
        sum(1 for w, c in Counter(w.lower() for w in alpha_words).items() if c == 1) / max(nw, 1),
    ]

    # LLM phrase patterns (15 features)
    for pat in LLM_PATTERNS.values():
        feats.append(min(len(re.findall(pat, tl)) / ns, 5.0))

    # Readability proxies (3 features)
    feats.append(np.mean(wl) / max(np.mean(sl), 1) if wl and sl else 0)
    feats.append(min(ns / max(nw/25, 1), 1.0))
    feats.append(uq / max(len(alpha_words), 1))

    arr = np.array(feats[:40], dtype=np.float32)
    return np.nan_to_num(arr, nan=0, posinf=5, neginf=0)


def extract_batch(texts, n_jobs=4):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return np.vstack(list(ex.map(extract_stylometric, texts)))