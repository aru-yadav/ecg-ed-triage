"""Pydantic Models"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np

class ModeInfo(BaseModel):
    name: str
    threshold: float
    sensitivity: float
    specificity: float
    use_case: str

class PredictionRequest(BaseModel):
    ecg_data: List[List[float]]
    mode: str = "high_safety"
    patient_id: Optional[str] = None

    @validator('ecg_data')
    def validate_ecg_shape(cls, v):
        arr = np.array(v)
        if arr.shape != (12, 1000):
            raise ValueError(f"ECG shape must be (12, 1000)")
        return v

class PredictionResponse(BaseModel):
    risk_category: str
    mi_probability: float
    mode: str
    mode_info: ModeInfo
    sensitivity_specificity: str
    recommendation: str
    priority: str
    confidence: str
    timestamp: str
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    auroc: float
    available_modes: List[str]
    uptime_seconds: float
    model_status: str
