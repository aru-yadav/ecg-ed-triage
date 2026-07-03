"""Readiness check:  PYTHONPATH=src python -m ecg_triage.check

Verifies Python, torch (+CUDA), core deps, the ensemble weights in models/, the
PTB-XL dataset, and the preprocessed cache, then prints a readiness summary.
Exit code 0 = ready to reproduce 0.9249, 1 = something is missing.
"""
import importlib
import platform
import sys

from . import data as D

MODEL_FILES = [f"ensemble_model_{i}.pt" for i in (1, 2, 3)]
CORE_DEPS = ["numpy", "pandas", "scipy", "sklearn", "wfdb"]


def _mark(ok: bool) -> str:
    return "OK  " if ok else "MISS"


def main():
    print("=" * 62)
    print("ecg_triage - readiness check")
    print("=" * 62)

    print(f"[OK  ] Python   : {platform.python_version()}")
    print(f"        exe      : {sys.executable}")

    try:
        import torch
        print(f"[OK  ] torch    : {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"[{'OK  ' if cuda else '--  '}] CUDA     : "
              f"{'available' if cuda else 'not available (CPU - fine, just slower)'}")
    except Exception as e:  # pragma: no cover
        print(f"[MISS] torch    : import failed ({e})")

    for mod in CORE_DEPS:
        try:
            m = importlib.import_module(mod)
            print(f"[OK  ] {mod:<9}: {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"[MISS] {mod:<9}: not importable")

    print("-" * 62)

    md = D.models_dir()
    missing_models = [f for f in MODEL_FILES if not (md / f).exists()]
    print(f"[{_mark(not missing_models)}] Models   : {md}")
    if missing_models:
        print(f"        missing  : {', '.join(missing_models)}")

    ds = D.dataset_dir()
    ds_ok = (ds / "ptbxl_database.csv").exists()
    print(f"[{_mark(ds_ok)}] Dataset  : {ds}")
    if not ds_ok:
        print(f"        get it   : {D.PTBXL_URL}")

    cp = D.cache_path()
    cache_ok = cp.exists()
    print(f"[{_mark(cache_ok)}] Cache    : {cp}")

    ready = (not missing_models) and (ds_ok or cache_ok)
    print("=" * 62)
    if ready:
        print("READY - run:  PYTHONPATH=src python -m ecg_triage.evaluate")
        print("expected:     TEST (fold 10) AUROC = 0.9249")
    else:
        print("NOT READY - need models present AND (dataset OR prebuilt cache).")
        print("see the missing items above.")
    print("=" * 62)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
