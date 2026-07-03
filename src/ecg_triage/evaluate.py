"""Evaluate the fold-10 ensemble that ships in models/.

- ensemble probability = mean over 3 models of softmax(logits)[:, 1]
- AUROC on val (fold 9) and test (fold 10)
- 3 clinical thresholds chosen on fold 9 ONLY (target sensitivities matching the
  named modes: max_safety>=0.98, high_safety>=0.96, balanced>=0.93), i.e. the
  highest threshold whose val-sensitivity still meets the target.
- sensitivity/specificity then reported on fold 10 at those frozen thresholds.

Weights load from models/ by default (the 3 files committed to this repo).
Override the weights dir with --models or the MODELS_DIR env var, and the dataset
dir with --data or PTBXL_DIR.

    PYTHONPATH=src python -m ecg_triage.evaluate
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from . import data as D
from .data import get_splits
from .model import ImprovedECGModel

TARGETS = [("max_safety", 0.98), ("high_safety", 0.96), ("balanced", 0.93)]
MODEL_FILES = [f"ensemble_model_{i}.pt" for i in (1, 2, 3)]


def require_models(model_dir: Path):
    """Raise a friendly error listing exactly which weight files are missing."""
    present = [f for f in MODEL_FILES if (model_dir / f).exists()]
    missing = [f for f in MODEL_FILES if f not in present]
    if missing:
        raise RuntimeError(
            "Ensemble weights not found.\n"
            f"  looked in : {model_dir}\n"
            f"  missing   : {', '.join(missing)}\n"
            f"  present   : {', '.join(present) if present else '(none)'}\n"
            "  This repo ships all 3 files in models/. If you moved them, pass\n"
            "  --models <dir> or set MODELS_DIR to point at the folder."
        )


def load_models(model_dir, device):
    models = []
    for f in MODEL_FILES:
        m = ImprovedECGModel(num_classes=2)
        m.load_state_dict(torch.load(model_dir / f, map_location=device))
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


def startup_checks(model_dir: Path, device):
    """Validate weights/dataset/cache up front and print an OK summary."""
    print("=" * 64)
    print("ecg_triage.evaluate - startup checks")
    print("-" * 64)

    require_models(model_dir)
    print(f"  [OK] Models   : {model_dir}  ({len(MODEL_FILES)} weight files)")

    cfg = model_dir / "ensemble_config.json"
    print(f"  [{'OK' if cfg.exists() else '--'}] Config   : "
          f"{cfg if cfg.exists() else '(ensemble_config.json not found - not required)'}")

    cp = D.cache_path()
    if cp.exists():
        print(f"  [OK] Cache    : {cp}")
        print(f"  [--] Dataset  : not needed (using prebuilt cache)")
    else:
        ds = D.require_dataset()
        print(f"  [OK] Dataset  : {ds}")
        print(f"  [..] Cache    : will be built at {cp}")

    print(f"  [OK] Device   : {device}")
    print("=" * 64, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproduce the fold-10 (patient-grouped) MI-detection metrics "
                    "from the 3-model ensemble in models/.")
    ap.add_argument("--data", metavar="DIR",
                    help="PTB-XL dataset dir (overrides PTBXL_DIR / default data/raw/...). "
                         "Ignored if the preprocessed cache already exists.")
    ap.add_argument("--models", metavar="DIR",
                    help="dir containing ensemble_model_{1,2,3}.pt "
                         "(overrides MODELS_DIR / default models/).")
    args = ap.parse_args(argv)

    if args.data:
        os.environ["PTBXL_DIR"] = args.data
    model_dir = Path(args.models) if args.models else D.models_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    startup_checks(model_dir, device)

    splits = get_splits()
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    models = load_models(model_dir, device)

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
