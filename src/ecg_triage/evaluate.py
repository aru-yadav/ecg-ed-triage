"""Evaluate the fold-CV ensemble.

- ensemble probability = mean over 3 models of softmax(logits)[:, 1]
- AUROC on val (fold 9) and test (fold 10)
- 3 clinical thresholds chosen on fold 9 ONLY (target sensitivities matching the
  named modes: max_safety>=0.98, high_safety>=0.96, balanced>=0.93), i.e. the
  highest threshold whose val-sensitivity still meets the target.
- sensitivity/specificity then reported on fold 10 at those frozen thresholds.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from .data import get_splits, REPO_ROOT
from .model import ImprovedECGModel

# Same override as train.py so evaluate reads the weights wherever they were saved.
MODEL_DIR = Path(os.environ.get("MODELS_FOLDCV_DIR", REPO_ROOT / "models_foldcv"))
TARGETS = [("max_safety", 0.98), ("high_safety", 0.96), ("balanced", 0.93)]


def load_models(device):
    models = []
    for i in range(1, 4):
        m = ImprovedECGModel(num_classes=2)
        m.load_state_dict(torch.load(MODEL_DIR / f"ensemble_model_{i}.pt", map_location=device))
        m.eval().to(device)
        models.append(m)
    return models


def ensemble_probs(models, X, device, batch=256):
    probs = np.zeros(len(X), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.FloatTensor(X[s:s+batch]).to(device)
            acc = np.zeros(len(xb), dtype=np.float64)
            for m in models:
                acc += F.softmax(m(xb), dim=1)[:, 1].cpu().numpy()
            probs[s:s+batch] = acc / len(models)
    return probs


def pick_threshold(y_val, p_val, target_sens):
    """Highest threshold whose val sensitivity >= target_sens."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = tpr >= target_sens
    # thr is sorted descending; among points meeting target, take the largest thr
    cand = thr[ok]
    return float(cand.max())


def sens_spec(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens * 100, spec * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = get_splits()
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    models = load_models(device)

    p_val = ensemble_probs(models, Xva, device)
    p_test = ensemble_probs(models, Xte, device)

    auroc_val = roc_auc_score(yva, p_val)
    auroc_test = roc_auc_score(yte, p_test)

    print("=" * 64)
    print(f"VAL  (fold 9)  AUROC = {auroc_val:.4f}   n={len(yva)} MI={int(yva.sum())}")
    print(f"TEST (fold 10) AUROC = {auroc_test:.4f}   n={len(yte)} MI={int(yte.sum())}")
    print("=" * 64)
    print(f"{'mode':<12}{'thr(val)':>10}{'tgt_sens':>10}  | {'TEST sens%':>11}{'TEST spec%':>11}")
    print("-" * 64)
    for name, target in TARGETS:
        thr = pick_threshold(yva, p_val, target)
        sens, spec = sens_spec(yte, p_test, thr)
        print(f"{name:<12}{thr:>10.4f}{target*100:>9.0f}%  | {sens:>10.2f}%{spec:>10.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
