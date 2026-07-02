# ECG ED Triage — Myocardial Infarction Detection from 12-Lead ECG

A research prototype that flags myocardial infarction (MI) from a standard
12-lead ECG, built as a decision-support aid for emergency-department triage.
A 3-model ResNet ensemble outputs an MI probability with three
selectable operating points (safety-first → screening).

> ⚠️ **Research prototype only.** This is **not** a medical device. It is **not**
> FDA/CE cleared and **must not** be used for clinical decision-making or patient
> care. It is provided for research and reproducibility purposes only.

## Headline result

**TEST (fold 10) AUROC = 0.9249** — PTB-XL, patient-grouped hold-out.

![ROC curve — MI detection on the PTB-XL fold-10 held-out test set (n=2198), AUROC = 0.9249, with the max_safety / high_safety / balanced operating points marked](docs/roc_fold10.png)

Evaluated with a strict, patient-grouped split using PTB-XL's official
`strat_fold` column (no patient appears in more than one fold):

| Split | Fold | n | MI | AUROC |
|-------|------|------|-----|--------|
| Train | 1–8  | —    | —   | —      |
| Val   | 9    | 2183 | 537 | 0.9247 |
| **Test**  | **10**   | **2198** | **550** | **0.9249** |

3-model ensemble (seeds 42 / 142 / 242), ensemble probability = mean of
per-model `softmax(logits)[MI]`.

> **Note on methodology.** Thresholds and all reported operating points are
> tuned on **fold 9 only** and measured on the untouched **fold 10** test set.
> Earlier random-split numbers (≈0.94) are **deprecated** — they were not
> patient-grouped and are not reported here. See [`RESULTS.txt`](RESULTS.txt)
> for the canonical, reproducible figures.

## Operating points

Thresholds tuned on fold 9, measured on fold 10:

| Mode | Threshold | Sensitivity | Specificity | Intended use |
|------|-----------|-------------|-------------|--------------|
| `max_safety`  | 0.1867 | 98.73% | 51.70% | ED, unstable / high-risk patients — minimize missed MI |
| `high_safety` | 0.2350 | 96.91% | 62.14% | ED standard triage (recommended default) |
| `balanced`    | 0.2975 | 93.82% | 71.54% | Outpatient screening |

## Reproduce in one command

1. Download PTB-XL v1.0.3 from PhysioNet (see **Data & attribution** below) into
   `data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
   (or point the `PTBXL_DIR` env var at your copy).
2. Install dependencies and run the evaluation:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m ecg_triage.evaluate
```

This loads `models/ensemble_model_{1,2,3}.pt`, prints VAL/TEST AUROC and the
three operating points, and reproduces [`RESULTS.txt`](RESULTS.txt). The first
run preprocesses PTB-XL and caches it to `data/processed/`; later runs reuse the
cache.

## Repository layout

```
src/ecg_triage/     Seeded, patient-grouped fold pipeline (canonical training code)
  data.py           PTB-XL loading, MI labels, strat_fold split (train 1-8 / val 9 / test 10)
  preprocess.py     Bandpass 0.5-40 Hz -> per-lead z-score -> 1000 samples
  model.py          ImprovedECGModel (ResNet-based 1D CNN)
  train.py          Train the 3-seed ensemble
  evaluate.py       Reproduce RESULTS.txt (AUROC + operating points)
api/                FastAPI inference service (prediction + Grad-CAM XAI + PDF reports)
models/             Canonical fold-10 weights + ensemble_config.json (operating points)
notebooks/          Exploratory notebooks (outputs stripped; see note below)
RESULTS.txt         Canonical, reproducible metrics — the source of truth
```

## Inference API

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload        # run from the repo root
```

The API loads the weights from `models/` relative to the repo root (override
with the `MODELS_DIR` env var). Endpoints:

- `GET  /`            — service info + current AUROC + available modes
- `GET  /healthz`     — readiness / loaded-model status
- `GET  /modes`       — the three operating-point definitions
- `POST /predict`     — MI probability + decision for a chosen mode
- `POST /explain`     — Grad-CAM lead-wise attribution + clinical statement

PDF triage reports can be generated programmatically via
`api.report_generator.generate_ecg_report(...)`.

## Data & attribution

This project uses **PTB-XL, a large publicly available electrocardiography
dataset**, distributed by PhysioNet under **CC BY 4.0**. The dataset is **not**
redistributed in this repository — download it directly from PhysioNet at
https://physionet.org/content/ptb-xl/1.0.3/.

**References**

1. Wagner et al. *PTB-XL, a large publicly available electrocardiography
   dataset.* Scientific Data 7, 154 (2020).
   https://doi.org/10.1038/s41597-020-0495-6
2. Goldberger et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
   Circulation 101(23):e215–e220 (2000).

MI labels use the PTB-XL SCP-ECG MI codes (`IMI, AMI, LMI, PMI, ASMI, ILMI,
ALMI, INJAS, INJAL, IPLMI, IPMI`).

## Notebooks

The notebooks are exploratory and provided for reference with their outputs
stripped. `notebooks/api/02_production_model.ipynb` documents earlier
**random-split** modeling and is **superseded** — the canonical, patient-grouped
pipeline lives in `src/ecg_triage/` and the canonical metrics are in
`RESULTS.txt`. Do not cite the notebooks' random-split numbers.

## License

Code is released under the [MIT License](LICENSE). The PTB-XL dataset and the
PTB-XL-derived model weights remain subject to PTB-XL's CC BY 4.0 terms. See the
[LICENSE](LICENSE) file for the full data/weights notice.
