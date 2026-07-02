"""Additive, seeded, fold-based retraining pipeline for ECG MI triage.

Independent of the existing notebooks / api / weights. Uses PTB-XL's
patient-grouped strat_fold column for splitting (train=1-8, val=9, test=10).
"""
