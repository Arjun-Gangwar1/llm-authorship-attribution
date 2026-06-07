# Unfreezing experiments

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time, os
from pathlib import Path


class LLMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(int(self.labels[i]), dtype=torch.long)}


def freeze_all_except_head(model):
    """Experiment 1: Only classifier head trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = 'classifier' in name or 'pooler' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen all except head. Trainable: {trainable:,}")
    return model


def unfreeze_top_n_layers(model, n_layers):
    """
    Experiments 2-4: Unfreeze top N encoder layers + head.
    DeBERTa-base has 12 transformer layers (layer.0 to layer.11).
    Unfreezing top N means layer.{12-n} to layer.11.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier + pooler
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            param.requires_grad = True

    # Unfreeze top N layers
    total_layers = 12  # DeBERTa-base
    for layer_idx in range(total_layers - n_layers, total_layers):
        for name, param in model.named_parameters():
            if f'layer.{layer_idx}.' in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfrozen top {n_layers} layers + head. Trainable: {trainable:,}")
    return model


def unfreeze_all(model):
    """Experiment 5: Full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Full fine-tuning. Trainable: {trainable:,}")
    return model


class TransformerTrainer:
    """
    Complete trainer for any HuggingFace sequence classification model.
    Supports all unfreezing strategies, LLRD, fp16, gradient accumulation.
    """

    UNFREEZE_STRATEGIES = {
        'head_only':    freeze_all_except_head,
        'top2':         lambda m: unfreeze_top_n_layers(m, 2),
        'top4':         lambda m: unfreeze_top_n_layers(m, 4),
        'top6':         lambda m: unfreeze_top_n_layers(m, 6),
        'full':         unfreeze_all,
    }

    def __init__(self, model_name, n_classes, strategy='full', device=None):
        self.model_name = model_name
        self.n_classes  = n_classes
        self.strategy   = strategy
        self.device     = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history    = []
        self.best_acc   = 0
        self.best_state = None

    def build(self):
        print(f"\n  Loading {self.model_name} with strategy '{self.strategy}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir='hf_cache')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_classes,
            cache_dir='hf_cache', ignore_mismatched_sizes=True
        )
        self.model = self.UNFREEZE_STRATEGIES[self.strategy](self.model)
        self.model = self.model.to(self.device)

        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)
        return self

    def _build_optimizer(self, n_steps, lr, warmup_ratio, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        # Layer-wise LR decay: classifier head gets 10× higher LR
        params = [
            {'params': [p for n, p in self.model.named_parameters()
                        if 'classifier' in n and p.requires_grad and not any(nd in n for nd in no_decay)],
             'lr': lr * 10, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters()
                        if 'classifier' not in n and p.requires_grad and not any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': 0.0},
        ]
        # Remove empty groups
        params = [g for g in params if g['params']]
        opt  = optim.AdamW(params, eps=1e-8)
        warm = int(n_steps * warmup_ratio)
        sch  = get_cosine_schedule_with_warmup(opt, warm, n_steps)
        return opt, sch

    def train(self, X_tr, y_tr, X_vl, y_vl,
              epochs=3, batch_size=16, grad_accum=4, lr=2e-5,
              warmup_ratio=0.1, weight_decay=0.01, label_smooth=0.05,
              max_len=512, save_path=None):

        tr_dl = DataLoader(
            LLMDataset(X_tr, y_tr, self.tokenizer, max_len),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        vl_dl = DataLoader(
            LLMDataset(X_vl, y_vl, self.tokenizer, max_len),
            batch_size=batch_size * 2, shuffle=False, num_workers=2
        )

        n_steps = (len(tr_dl) // grad_accum) * epochs
        opt, sch = self._build_optimizer(n_steps, lr, warmup_ratio, weight_decay)
        crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        use_fp16 = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

        print(f"\n  Training: {epochs} epochs | eff_batch={batch_size*grad_accum} | lr={lr} | fp16={use_fp16}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            opt.zero_grad()

            for step, batch in enumerate(tr_dl):
                ids  = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                lbls = batch['label'].to(self.device)

                if use_fp16:
                    with torch.cuda.amp.autocast():
                        out  = self.model(input_ids=ids, attention_mask=mask)
                        loss = crit(out.logits, lbls) / grad_accum
                    scaler.scale(loss).backward()
                else:
                    out  = self.model(input_ids=ids, attention_mask=mask)
                    loss = crit(out.logits, lbls) / grad_accum
                    loss.backward()

                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    if use_fp16:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if use_fp16:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    sch.step(); opt.zero_grad()

            val_acc, val_f1 = self._evaluate(vl_dl)
            avg_loss = total_loss / len(tr_dl)
            self.history.append({'epoch': epoch, 'loss': avg_loss,
                                  'val_acc': val_acc, 'val_f1': val_f1})
            print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}", end='')

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                raw = self.model.module if hasattr(self.model, 'module') else self.model
                self.best_state = {k: v.clone().cpu() for k, v in raw.state_dict().items()}
                if save_path:
                    torch.save({'state': self.best_state, 'model_name': self.model_name,
                                'n_classes': self.n_classes, 'strategy': self.strategy},
                               save_path)
                print("  ★ best")
            else:
                print()

        # Restore best
        raw = self.model.module if hasattr(self.model, 'module') else self.model
        raw.load_state_dict(self.best_state)
        return self

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        preds, trues = [], []
        for batch in loader:
            out = self.model(input_ids=batch['input_ids'].to(self.device),
                             attention_mask=batch['attention_mask'].to(self.device))
            preds.extend(out.logits.argmax(1).cpu().numpy())
            trues.extend(batch['label'].numpy())
        return accuracy_score(trues, preds), f1_score(trues, preds, average='macro')

    @torch.no_grad()
    def predict_proba(self, loader):
        import torch.nn.functional as F
        self.model.eval()
        probs = []
        for batch in loader:
            out = self.model(input_ids=batch['input_ids'].to(self.device),
                             attention_mask=batch['attention_mask'].to(self.device))
            probs.append(F.softmax(out.logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def full_report(self, X_ts, y_ts, class_names, max_len=512, batch_size=32):
        ts_dl = DataLoader(
            LLMDataset(X_ts, y_ts, self.tokenizer, max_len),
            batch_size=batch_size, shuffle=False, num_workers=2
        )
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in ts_dl:
                out = self.model(input_ids=batch['input_ids'].to(self.device),
                                 attention_mask=batch['attention_mask'].to(self.device))
                preds.extend(out.logits.argmax(1).cpu().numpy())
                trues.extend(batch['label'].numpy())
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='macro')
        print(f"\n  Test Acc: {acc:.4f}  F1: {f1:.4f}")
        print(classification_report(trues, preds, target_names=class_names, digits=4))
        return acc, f1