"""Configuration Module.

Metrics and thresholds are the canonical fold-10 (patient-grouped) results:
PTB-XL strat_fold split train 1-8 / val 9 / test 10, seeds 42/142/242.
See RESULTS.txt and models/ensemble_config.json.
"""

import os
from pathlib import Path
from typing import Dict

VERSION = "2.0.0"
AUROC = 0.9249  # TEST (fold 10) AUROC — canonical, patient-grouped hold-out
CONFIDENCE_THRESHOLD = 0.15
ABSTENTION_THRESHOLD = 0.10

MODES = {
    "max_safety": {
        "name": "Maximum Safety Mode",
        "threshold": 0.1867,
        "sensitivity": 98.73,
        "specificity": 51.70,
        "use_case": "Emergency Department - High-Risk Patients"
    },
    "high_safety": {
        "name": "High Safety Mode (RECOMMENDED)",
        "threshold": 0.2350,
        "sensitivity": 96.91,
        "specificity": 62.14,
        "use_case": "Emergency Department - Standard Triage"
    },
    "balanced": {
        "name": "Balanced Mode",
        "threshold": 0.2975,
        "sensitivity": 93.82,
        "specificity": 71.54,
        "use_case": "Outpatient Clinic - Screening"
    }
}

# Resolve model dir relative to the repo root (this file: api/config.py) so the
# API loads weights regardless of the current working directory. Override with
# the MODELS_DIR env var. Fixes the CWD-relative "../models" bug.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))
MODEL_FILES = [f"ensemble_model_{i}.pt" for i in range(1, 4)]
API_TITLE = "ECG MI Detection API"
API_DESCRIPTION = "Research prototype: 3-mode ensemble MI triage on 12-lead ECG"
LOG_FILE = "api_audit.log"
