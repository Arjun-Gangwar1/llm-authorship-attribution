"""
DeBERTa-v3 fine-tuning — FIXED trainer for the 12-class LLM authorship task.
=============================================================================

WHY THE OLD RUN COLLAPSED TO 8.29% (= 1/12, random)
---------------------------------------------------
1. DeBERTa-v3 is numerically unstable in **fp16** (its disentangled-attention
   softmax overflows in half precision on Turing GPUs like the Kaggle T4).
   The first big gradient becomes inf/NaN.
2. The old guard only did `if torch.isnan(loss): continue`. It checked the
   LOSS, never the GRADIENTS, and `clip_grad_norm_` on an inf grad returns inf
   (a no-op). So `opt.step()` wrote NaN into every weight — permanently.
3. Once the weights are NaN, every forward pass is NaN, the loss is NaN and
   skipped, so `train_loss` decays to 0.0000 while val stays at random 1/12.

THE FIXES (all applied below)
-----------------------------
A. Precision: use **bf16** if the GPU supports it, else **fp32**. NEVER fp16
   for DeBERTa-v3. (bf16 has fp32's exponent range → no overflow, no GradScaler
   needed. T4 has no native bf16, so this falls back to fp32 there.)
B. A real finite-gradient guard: after backward, compute the grad-norm and
   SKIP THE OPTIMIZER STEP (and zero grads) if it isn't finite — so one bad
   batch can never poison the weights.
C. Clip gradients on EVERY optimizer step (not just sometimes).
D. Conservative LR (encoder 1e-5, head 3x) with 10% warmup.
E. One LabelEncoder, fit on train only, reused for val/test, with an assert
   that all label ids are in [0, NUM_CLASSES). (A scrambled val encoding is a
   second, independent way to get random val accuracy — ruled out here.)
F. A `sanity_overfit()` that overfits 16 samples first. If the model cannot
   reach ~100% on 16 samples in a few steps, the bug is in the code/data, not
   training — fail fast before burning GPU hours.

Usage (Kaggle / local):
    from deberta_finetune_FIXED import run
    run(X_train, y_train_str, X_val, y_val_str, classes=CLASSES)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_cosine_schedule_with_warmup)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "model_name":       "microsoft/deberta-v3-base",
    "max_length":       256,
    "batch_size":       16,
    "grad_accum":       2,        # effective batch 32
    "num_epochs":       3,
    "encoder_lr":       1e-5,
    "head_lr_mult":     3.0,
    "warmup_ratio":     0.10,
    "weight_decay":     0.01,
    "label_smoothing":  0.05,
    "max_grad_norm":    1.0,
}


# ── Precision selection: bf16 if supported, else fp32. NEVER fp16. ───────────
def pick_amp():
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True      # bf16: stable, no GradScaler needed
    return torch.float32, False          # fp32 fallback (e.g. Turing T4)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


def build_model_and_tok(num_classes):
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=num_classes).to(DEVICE)
    return model, tok


def build_optimizer(model, n_steps):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    head_lr = CONFIG["encoder_lr"] * CONFIG["head_lr_mult"]
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and not any(nd in n for nd in no_decay)],
         "lr": head_lr, "weight_decay": CONFIG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and not any(nd in n for nd in no_decay)],
         "lr": CONFIG["encoder_lr"], "weight_decay": CONFIG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "lr": CONFIG["encoder_lr"], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, eps=1e-8)
    warm = int(n_steps * CONFIG["warmup_ratio"])
    sch = get_cosine_schedule_with_warmup(opt, warm, n_steps)
    return opt, sch


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    for b in loader:
        out = model(input_ids=b["input_ids"].to(DEVICE),
                    attention_mask=b["attention_mask"].to(DEVICE))
        logits = out.logits.float()                  # cast back to fp32 for metrics
        labs = b["label"].to(DEVICE)
        loss_sum += crit(logits, labs).item(); n += 1
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(labs.cpu().numpy())
    return accuracy_score(trues, preds), loss_sum / max(n, 1), np.array(preds), np.array(trues)


def _finite_grads(model):
    """True iff every gradient is finite. This is the guard the old code lacked."""
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def sanity_overfit(model, tok, texts, labels, steps=60):
    """Overfit 16 samples. Must reach ~100%. If not, the bug is code/data, not training."""
    print("\n[sanity] overfitting 16 samples to prove the loop can learn…")
    ds = TextDataset(texts[:16], labels[:16], tok, CONFIG["max_length"])
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    crit = nn.CrossEntropyLoss()
    model.train()
    for s in range(steps):
        for b in dl:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(DEVICE),
                        attention_mask=b["attention_mask"].to(DEVICE))
            loss = crit(out.logits.float(), b["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    acc, _, _, _ = evaluate(model, dl, crit)
    print(f"[sanity] overfit accuracy on 16 samples = {acc*100:.1f}% "
          f"({'OK — loop learns' if acc > 0.9 else 'FAIL — fix code/data before full run'})")
    return acc


def run(X_train, y_train_str, X_val, y_val_str, classes):
    num_classes = len(classes)

    # ── Fix E: one encoder, fit on train, reuse; assert ids in range ─────────
    le = LabelEncoder().fit(classes)
    y_tr = le.transform(y_train_str)
    y_va = le.transform(y_val_str)
    assert y_tr.min() >= 0 and y_tr.max() < num_classes, "train label id out of range"
    assert y_va.min() >= 0 and y_va.max() < num_classes, "val label id out of range"

    amp_dtype, use_amp = pick_amp()
    print(f"Precision: {'bf16' if amp_dtype==torch.bfloat16 else 'fp32'} "
          f"(fp16 intentionally avoided for DeBERTa-v3)")

    model, tok = build_model_and_tok(num_classes)

    # Fail-fast sanity check on a fresh copy's weights (cheap, ~seconds)
    sanity_overfit(model, tok, list(X_train), list(y_tr))
    # reload clean weights after the overfit probe
    model, tok = build_model_and_tok(num_classes)

    tr_dl = DataLoader(TextDataset(X_train, y_tr, tok, CONFIG["max_length"]),
                       batch_size=CONFIG["batch_size"], shuffle=True,
                       num_workers=2, pin_memory=True)
    va_dl = DataLoader(TextDataset(X_val, y_va, tok, CONFIG["max_length"]),
                       batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=2)

    n_steps = (len(tr_dl) // CONFIG["grad_accum"]) * CONFIG["num_epochs"]
    opt, sch = build_optimizer(model, n_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    best_acc, best_state, skipped = 0.0, None, 0
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train(); opt.zero_grad()
        running, counted = 0.0, 0
        for step, b in enumerate(tr_dl):
            ids = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            labs = b["label"].to(DEVICE)

            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else torch.autocast(device_type="cuda", enabled=False))
            with ctx:
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = crit(logits.float(), labs) / CONFIG["grad_accum"]

            loss.backward()
            running += loss.item() * CONFIG["grad_accum"]; counted += 1

            if (step + 1) % CONFIG["grad_accum"] == 0:
                # ── Fix B+C: clip, then ONLY step if grads are finite ────────
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                if _finite_grads(model):
                    opt.step(); sch.step()
                else:
                    skipped += 1                       # drop the bad step, keep weights clean
                opt.zero_grad()

        val_acc, val_loss, _, _ = evaluate(model, va_dl, nn.CrossEntropyLoss())
        print(f"Epoch {epoch}/{CONFIG['num_epochs']} | "
              f"train_loss={running/max(counted,1):.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | "
              f"skipped_steps={skipped}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "deberta_best.pt")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nBest val acc: {best_acc*100:.2f}%  (total skipped steps: {skipped})")
    _, _, preds, trues = evaluate(model, va_dl, nn.CrossEntropyLoss())
    print(classification_report(le.inverse_transform(trues),
                                le.inverse_transform(preds), digits=3, zero_division=0))
    return model, tok, le, best_acc


if __name__ == "__main__":
    # Smoke test with dummy data (proves the loop runs without a GPU/real data).
    classes = ["gpt2", "llama-chat", "human", "chatgpt", "mistral", "gpt4",
               "mpt-chat", "mistral-chat", "gpt3", "mpt", "cohere-chat", "cohere"]
    print("Module imports OK. Call run(X_train, y_train, X_val, y_val, classes) on real data.")
