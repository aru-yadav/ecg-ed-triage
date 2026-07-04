"""Preprocessing — ported VERBATIM from 02_production_model.ipynb (cell 3).

Bandpass (0.5-40 Hz, butter order 4) -> per-lead z-score -> fixed 1000 samples.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def preprocess_ecg_advanced(ecg, orig_fs=100, target_fs=100):
    """Advanced preprocessing with robust filtering (verbatim from notebook)."""
    # Ensure shape is (12, T)
    if ecg.shape[0] != 12:
        ecg = ecg.T

    # 1. Bandpass filter (0.5-40 Hz)
    nyq = 0.5 * orig_fs
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype='band')
    ecg_filtered = filtfilt(b, a, ecg, axis=1)

    # 2. Z-score normalization per lead. Clamp the denominator so a flat/constant
    # lead (std -> 0) can never produce a divide-by-~0 -> NaN; it normalizes to 0.
    mean = ecg_filtered.mean(axis=1, keepdims=True)
    std = np.maximum(ecg_filtered.std(axis=1, keepdims=True), 1e-8)
    ecg_norm = (ecg_filtered - mean) / std

    # 3. Ensure fixed length (1000 samples)
    if ecg_norm.shape[1] != 1000:
        if ecg_norm.shape[1] > 1000:
            ecg_norm = ecg_norm[:, :1000]
        else:
            pad_width = 1000 - ecg_norm.shape[1]
            ecg_norm = np.pad(ecg_norm, ((0, 0), (0, pad_width)), mode='edge')

    return ecg_norm.astype(np.float32)
