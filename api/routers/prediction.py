"""Prediction Router"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
import logging
import numpy as np

from api.models import PredictionRequest, PredictionResponse, ModeInfo
from api.config import VERSION, MODES
from api.core.safety import SafetyGuardrails
from api.core.logging_config import log_prediction, log_abstention

router = APIRouter()
logger = logging.getLogger("ecg_api")
safety = SafetyGuardrails()
predictor = None

def set_predictor(pred):
    global predictor
    predictor = pred

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_ms = datetime.now().timestamp() * 1000
    timestamp = datetime.now().isoformat()

    try:
        ecg_array = np.array(request.ecg_data, dtype=np.float32)
        result = predictor.predict(ecg_array, request.mode)

        if not result["success"]:
            return PredictionResponse(
                risk_category="UNABLE_TO_CLASSIFY",
                mi_probability=0.5,
                mode="max_safety",
                mode_info=ModeInfo(**MODES["max_safety"]),
                sensitivity_specificity="N/A",
                recommendation="SYSTEM ERROR - Clinical review required.",
                priority="URGENT",
                confidence="SYSTEM_ERROR",
                timestamp=timestamp,
                model_version=VERSION,
                processing_time_ms=(datetime.now().timestamp() * 1000) - start_ms
            )

        if result["abstained"]:
            risk_cat, rec, pri = safety.generate_recommendation(
                "ABSTAINED", result["probability"], result["mode"], "VERY_LOW", True
            )
            return PredictionResponse(
                risk_category=risk_cat,
                mi_probability=round(result["probability"], 4),
                mode=result["mode"],
                mode_info=ModeInfo(**result["mode_config"]),
                sensitivity_specificity="N/A",
                recommendation=rec,
                priority=pri,
                confidence="ABSTAINED",
                timestamp=timestamp,
                model_version=VERSION,
                processing_time_ms=round(result["processing_time_ms"], 2)
            )

        conf_level = safety.classify_confidence(result["confidence_score"])
        risk_cat, rec, pri = safety.generate_recommendation(
            result["prediction"], result["probability"], result["mode"], conf_level
        )

        return PredictionResponse(
            risk_category=risk_cat,
            mi_probability=round(result["probability"], 4),
            mode=result["mode"],
            mode_info=ModeInfo(**result["mode_config"]),
            sensitivity_specificity=f"{result['mode_config']['sensitivity']}% / {result['mode_config']['specificity']}%",
            recommendation=rec,
            priority=pri,
            confidence=conf_level,
            timestamp=timestamp,
            model_version=VERSION,
            processing_time_ms=round((datetime.now().timestamp() * 1000) - start_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
