"""
Run after all experiments complete.
Reads all JSON result files and generates:
- Per-embedding comparison table (all classifiers)
- Global comparison table (all models)
- LaTeX table for paper/report
"""
import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("reports/figures"); FIGURES_DIR.mkdir(exist_ok=True)

def load_all_results():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
            rows.append(d)
    return pd.DataFrame(rows)


def make_per_embedding_table(df):
    """One table per embedding, showing all classifiers."""
    embeddings = df['embedding'].unique() if 'embedding' in df.columns else ['all']
    tables = {}
    for emb in embeddings:
        sub = df[df['embedding'] == emb] if 'embedding' in df.columns else df
        pivot = sub[['classifier','val_acc','test_acc','val_f1','test_f1']].copy()
        pivot = pivot.sort_values('test_acc', ascending=False).reset_index(drop=True)
        tables[emb] = pivot
        print(f"\n=== {emb} ===")
        print(pivot.to_string(index=False))
    return tables


def plot_overall_comparison(df, save_path):
    df_sorted = df.sort_values('test_acc', ascending=True)
    fig, ax = plt.subplots(figsize=(14, max(6, len(df)*0.4)))
    colors = ['#2ecc71' if 'DeBERTa' in str(n) or 'Ensemble' in str(n)
              else '#3498db' if 'SBERT' in str(n) or 'BGE' in str(n) or 'E5' in str(n)
              else '#e67e22'
              for n in df_sorted['experiment']]
    bars = ax.barh(df_sorted['experiment'], df_sorted['test_acc'],
                   color=colors, edgecolor='white', height=0.7)
    for bar, acc in zip(bars, df_sorted['test_acc']):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', va='center', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.90, color='orange', linestyle='--', alpha=0.4, label='90%')
    ax.axvline(0.95, color='red',    linestyle='--', alpha=0.4, label='95%')
    ax.set_xlabel('Test Accuracy'); ax.set_title('All Models — Test Accuracy Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.close()


def make_latex_table(df, save_path):
    cols = ['experiment','train_acc','val_acc','test_acc','val_f1','test_f1']
    available = [c for c in cols if c in df.columns]
    latex = df[available].sort_values('test_acc', ascending=False).to_latex(
        index=False, float_format='%.4f',
        caption='LLM Text Classification Results',
        label='tab:results'
    )
    with open(save_path, 'w') as f: f.write(latex)
    print(f"LaTeX table saved → {save_path}")


if __name__ == '__main__':
    df = load_all_results()
    print(f"Loaded {len(df)} experiment results")
    make_per_embedding_table(df)
    plot_overall_comparison(df, FIGURES_DIR / 'overall_comparison.png')
    make_latex_table(df, REPORTS_DIR / 'tables/results_table.tex')
    df.to_csv(REPORTS_DIR / 'tables/all_results.csv', index=False)
    df.to_excel(REPORTS_DIR / 'tables/all_results.xlsx', index=False)
    print("\nDone. All reports generated.")