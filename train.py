"""Train and evaluate the multimodal detector.

Usage:
  python train.py --data data/dataset.csv                 # stacking fusion (default)
  python train.py --data data/dataset.csv --model neural  # attention-fusion net
  python train.py --data data/dataset.csv --ocr           # also OCR images (needs Tesseract)
  python train.py --data data/real_dataset.csv            # real data (template-grouped split)
"""
import argparse

import scamdetect  # noqa: F401  (sets LOKY_MAX_CPU_COUNT before sklearn loads)
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from scamdetect.fusion import MultimodalScamDetector, MODALITIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--model", choices=["stack", "neural"], default="stack")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.data, keep_default_na=False)
    groups = df["group"] if "group" in df else None
    if groups is not None:  # keep near-duplicate templates on one side of the split
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(df, groups=groups))
        tr, te = df.iloc[tr_i], df.iloc[te_i]
    else:
        tr, te = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=0)

    if a.model == "neural":
        from scamdetect.fusion_nn import NeuralScamDetector
        model = NeuralScamDetector(use_ocr=a.ocr)
    else:
        model = MultimodalScamDetector(use_ocr=a.ocr)
    print(f"training {a.model} model on {len(tr)} samples...")
    model.fit(tr, tr["label"], groups=None if groups is None else tr["group"])

    p = model.predict_proba(te)
    y = te["label"].to_numpy()
    print("\n=== Fused model (test set) ===")
    print(classification_report(y, (p >= 0.5).astype(int), target_names=["legit", "scam"], digits=3))
    print(f"ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f}")

    if "source" in te:
        print("\n=== By source (test) ===")
        for src, g in te.assign(p=p).groupby("source"):
            yy, pp = g["label"].to_numpy(), g["p"].to_numpy()
            auc = f"AUC {roc_auc_score(yy, pp):.3f}" if len(np.unique(yy)) == 2 else "AUC   n/a"
            print(f"{src:>13}: {auc} | acc {((pp >= 0.5) == yy).mean():.3f} | n={len(g)}")
    if "is_investment" in te:
        inv = (te["label"].to_numpy() == 1) & (te["is_investment"].astype(int).to_numpy() == 1)
        print(f"recall on investment-related scams: {(p[inv] >= 0.5).mean():.3f} (n={inv.sum()})")
        legit = y == 0
        print(f"false-positive rate on legit messages: {(p[legit] >= 0.5).mean():.3f} (n={legit.sum()})")

    if a.model == "stack":
        probs, present, _, _ = model.modality_probs(te)
        print("\n=== Single-modality experts (test, where available) ===")
        for j, m in enumerate(MODALITIES):
            mask = present[:, j]
            if mask.sum() and len(np.unique(y[mask])) == 2:
                print(f"{m:>6}: AUC {roc_auc_score(y[mask], probs[mask, j]):.3f}  (n={mask.sum()})")
        w = model.fuser_.coef_[0]
        print("fuser weights", {m: round(float(v), 2) for m, v in zip(MODALITIES, w)})
    else:
        _, att = model.predict_with_attention(te)
        print("mean modality attention:", pd.DataFrame(att).mean().round(3).to_dict())

    out = a.out or f"models/scam_detector_{a.model}.joblib"
    model.fit(df, df["label"], groups=groups)  # refit on all data for deployment
    model.save(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
