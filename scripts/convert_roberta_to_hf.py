"""
Convert a trained roberta_best.pt (state_dict) into a Hugging Face model folder.

Produces a folder loadable with:
    AutoModelForSequenceClassification.from_pretrained(OUT_DIR)

Usage:
    python scripts/convert_roberta_to_hf.py <path_to_roberta_best.pt> <output_dir>
"""
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The 12 classes in the EXACT order the LabelEncoder assigned them.
# LabelEncoder sorts alphabetically; this list is already alphabetical,
# so label id i  ==  CLASSES[i].
CLASSES = ['chatgpt', 'cohere', 'cohere-chat', 'gpt2', 'gpt3', 'gpt4',
           'human', 'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat']

BASE_MODEL = "roberta-base"


def main(pt_path: str, out_dir: str):
    id2label = {i: c for i, c in enumerate(CLASSES)}
    label2id = {c: i for i, c in enumerate(CLASSES)}

    print(f"1/4  Building {BASE_MODEL} architecture (12-class head)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(CLASSES),
        id2label=id2label, label2id=label2id,
    )

    print(f"2/4  Loading trained weights from {pt_path} ...")
    state = torch.load(pt_path, map_location="cpu")
    # handle a possible DataParallel "module." prefix (not expected here, but safe)
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"     missing keys: {len(missing)}  | unexpected keys: {len(unexpected)}")
    if missing:
        print(f"     (missing example: {missing[:3]})")
    if unexpected:
        print(f"     (unexpected example: {unexpected[:3]})")

    print(f"3/4  Saving tokenizer + model to {out_dir} ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    print("4/4  Done. Files written:")
    import os
    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f)) / 1024 / 1024
        print(f"       {f:32s} {sz:7.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_roberta_to_hf.py <pt_path> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
