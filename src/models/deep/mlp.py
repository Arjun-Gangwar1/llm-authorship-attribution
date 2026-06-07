#  Multi-head fusion model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class MultiHeadFusion(nn.Module):
    """
    Fusion model: separate branches for embeddings and stylometric features,
    then concatenated through a shared classifier.
    Mimics multi-modal learning applied to style signals.
    """
    def __init__(self, emb_dim, stylo_dim, n_classes=12, dropout=0.35):
        super().__init__()
        # Embedding branch
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.8),
        )
        # Stylometric branch
        self.stylo_branch = nn.Sequential(
            nn.Linear(stylo_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        # Fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.6),
            nn.Linear(256, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_emb, x_stylo):
        e = self.emb_branch(x_emb)
        s = self.stylo_branch(x_stylo)
        return self.fusion(torch.cat([e, s], dim=1))


def train_multihead_fusion(X_emb_tr, X_stylo_tr, y_tr,
                           X_emb_vl, X_stylo_vl, y_vl,
                           X_emb_ts, X_stylo_ts, y_ts,
                           epochs=80, lr=3e-4, batch_size=512,
                           device=None, save_path=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Xe_tr = torch.FloatTensor(X_emb_tr).to(device)
    Xs_tr = torch.FloatTensor(X_stylo_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    Xe_vl = torch.FloatTensor(X_emb_vl).to(device)
    Xs_vl = torch.FloatTensor(X_stylo_vl).to(device)
    Xe_ts = torch.FloatTensor(X_emb_ts).to(device)
    Xs_ts = torch.FloatTensor(X_stylo_ts).to(device)

    ds = TensorDataset(Xe_tr, Xs_tr, y_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MultiHeadFusion(X_emb_tr.shape[1], X_stylo_tr.shape[1]).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = len(dl) * epochs
    sch   = get_cosine_schedule_with_warmup(opt, n_steps // 10, n_steps)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for Xeb, Xsb, yb in dl:
            opt.zero_grad()
            loss = crit(model(Xeb, Xsb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl_preds = model(Xe_vl, Xs_vl).argmax(1).cpu().numpy()
            vl_acc = accuracy_score(y_vl, vl_preds)
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch:3d}  val_acc={vl_acc:.4f}  best={best_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ts_preds = model(Xe_ts, Xs_ts).argmax(1).cpu().numpy()
    test_acc = accuracy_score(y_ts, ts_preds)
    test_f1  = f1_score(y_ts, ts_preds, average='macro')
    print(f"\n  Multi-head Fusion  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

    if save_path:
        torch.save({'state': best_state, 'emb_dim': X_emb_tr.shape[1],
                    'stylo_dim': X_stylo_tr.shape[1]}, save_path)

    return model, {'val_acc': best_acc, 'test_acc': test_acc, 'test_f1': test_f1}