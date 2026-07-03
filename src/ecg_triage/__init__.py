"""Seeded, patient-grouped fold pipeline for ECG MI triage.

Splits PTB-XL by its strat_fold column (train = folds 1-8, val = fold 9,
test = fold 10) — no train_test_split, no fold mixing. The 3-seed ensemble
weights live in models/; evaluate.py reproduces the canonical fold-10 metrics.
"""
