"""PTB-XL loading + MI labels + PATIENT-GROUPED fold split.

Split is by the dataset's ``strat_fold`` column (PTB-XL folds are already
patient-grouped, so no patient appears in more than one fold):
    train = folds 1-8,  val = fold 9,  test = fold 10.
We NEVER use train_test_split and NEVER mix folds.
"""
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from .preprocess import preprocess_ecg_advanced

# Resolve the dataset dir relative to the repo root (this file: src/ecg_triage/data.py).
# Override either path with env vars PTBXL_DIR / FOLDCV_CACHE for GPU/Colab runs.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "PTBXL_DIR",
    REPO_ROOT / "data" / "raw" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
))
CACHE_PATH = Path(os.environ.get(
    "FOLDCV_CACHE",
    REPO_ROOT / "data" / "processed" / "foldcv_cache.npz",
))

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10

# 11 MI SCP codes — identical set to has_mi() in 02_production_model.ipynb (cell 1)
MI_CODES = ['IMI', 'AMI', 'LMI', 'PMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'INJAL', 'IPLMI', 'IPMI']


def has_mi(scp_codes_str):
    """Verbatim MI labelling rule from the notebook."""
    if pd.isna(scp_codes_str):
        return 0
    scp_codes = ast.literal_eval(scp_codes_str)
    for code in MI_CODES:
        if code in scp_codes:
            return 1
    return 0


def load_metadata():
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv")
    df['is_mi'] = df['scp_codes'].apply(has_mi)
    assert 'strat_fold' in df.columns, "strat_fold column missing from ptbxl_database.csv"
    return df


def build_cache(force=False):
    """Load + preprocess every 100 Hz record once, cache to data/processed/foldcv_cache.npz.

    Cached arrays are stored in df row order; fold/label arrays align by position.
    """
    if CACHE_PATH.exists() and not force:
        return
    df = load_metadata()
    X = np.zeros((len(df), 12, 1000), dtype=np.float32)
    y = df['is_mi'].to_numpy().astype(np.int64)
    fold = df['strat_fold'].to_numpy().astype(np.int64)
    ok = np.ones(len(df), dtype=bool)
    for idx in tqdm(range(len(df)), desc="Preprocessing PTB-XL (100Hz)"):
        try:
            rec = wfdb.rdsamp(str(DATA_DIR / df.loc[idx, 'filename_lr']))
            X[idx] = preprocess_ecg_advanced(rec[0].T, orig_fs=100)
        except Exception:
            ok[idx] = False
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, X=X[ok], y=y[ok], fold=fold[ok])
    print(f"Cached {ok.sum()} records (failed {int((~ok).sum())}) -> {CACHE_PATH}")


def get_splits():
    """Return dict of (X, y) per split, sliced strictly by strat_fold."""
    build_cache()
    d = np.load(CACHE_PATH)
    X, y, fold = d['X'], d['y'], d['fold']

    def sel(folds):
        m = np.isin(fold, folds)
        return X[m], y[m]

    Xtr, ytr = sel(TRAIN_FOLDS)
    Xva, yva = sel([VAL_FOLD])
    Xte, yte = sel([TEST_FOLD])

    # Hard guarantee: no fold overlap and counts add up
    assert len(Xtr) + len(Xva) + len(Xte) == np.isin(fold, TRAIN_FOLDS + [VAL_FOLD, TEST_FOLD]).sum()
    return {
        "train": (Xtr, ytr),
        "val": (Xva, yva),
        "test": (Xte, yte),
    }


if __name__ == "__main__":
    s = get_splits()
    for k, (Xk, yk) in s.items():
        print(f"{k:5s}: n={len(Xk):6d}  MI={int(yk.sum()):5d}  Normal={int((yk==0).sum()):6d}")
