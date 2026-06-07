# complete evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import json
from pathlib import Path


CLASSES = ['gpt2','llama-chat','human','chatgpt','mistral','gpt4',
           'mpt-chat','mistral-chat','gpt3','mpt','cohere-chat','cohere']


def compute_all_metrics(y_true, y_pred, y_proba=None, class_names=CLASSES):
    """Compute all evaluation metrics. Returns dict."""
    metrics = {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1_macro':  float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro':    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc_ovr'] = float(
                roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            )
            metrics['log_loss'] = float(log_loss(y_true, y_proba))
        except Exception as e:
            print(f"  Warning: {e}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, save_path,
                          class_names=CLASSES, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1 if normalize else None, linewidths=0.3)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_calibration(y_true, y_proba, model_name, save_path, n_bins=10):
    n_cls = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_cls):
        binary = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(binary, y_proba[:, i], n_bins=n_bins)
        ax.plot(prob_pred, prob_true, alpha=0.5, label=CLASSES[i] if i < len(CLASSES) else str(i))
    ax.plot([0,1],[0,1],'k--', label='Perfect')
    ax.set_title(f'Calibration Plot — {model_name}')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_results_table(results_dir='experiments/results'):
    """Load all JSON result files and build a comparison DataFrame."""
    rows = []
    for f in Path(results_dir).glob('*.json'):
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows).sort_values('val_acc', ascending=False)
    return df


def save_results(results_dict, path):
    with open(path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in results_dict.items()}, f, indent=2)