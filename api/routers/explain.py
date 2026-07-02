"""Explainability Router"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
import logging
import numpy as np

from api.models import PredictionRequest
from api.clinical.statement_generator import generate_clinical_statement
from pydantic import BaseModel
from typing import Dict, List

router = APIRouter()
logger = logging.getLogger("ecg_api")

predictor = None
explainer = None

def set_dependencies(pred, expl):
    global predictor, explainer
    predictor = pred
    explainer = expl

class ExplainResponse(BaseModel):
    patient_id: str
    mi_probability: float
    lead_importance: Dict[str, float]
    cardiac_regions: Dict[str, float]
    temporal_importance: Dict[str, float]
    top_contributing_leads: List[Dict]
    clinical_interpretation: str
    explanation_method: str
    timestamp: str

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: PredictionRequest):
    timestamp = datetime.now().isoformat()

    try:
        ecg_array = np.array(request.ecg_data, dtype=np.float32)
        result = predictor.predict(ecg_array, request.mode)

        if not result["success"]:
            raise HTTPException(status_code=500, detail="Prediction failed")

        if result.get("abstained", False):
            raise HTTPException(status_code=400, detail="Model abstained - explanation unavailable")

        explanation = explainer.explain(ecg_array, result["probability"])

        # Generate rule-based clinical statement
        top_leads = [lead["lead"] for lead in explanation["top_contributing_leads"]]
        clinical_statement = generate_clinical_statement(top_leads, explanation["cardiac_regions"])

        logger.info(f"AUDIT | EXPLAIN | patient={request.patient_id} | prob={result['probability']:.3f}")

        return ExplainResponse(
            patient_id=request.patient_id or "UNKNOWN",
            mi_probability=round(result["probability"], 4),
            lead_importance=explanation["lead_importance"],
            cardiac_regions=explanation["cardiac_regions"],
            temporal_importance=explanation["temporal_importance"],
            top_contributing_leads=explanation["top_contributing_leads"],
            clinical_interpretation=clinical_statement,
            explanation_method=explanation["explanation_method"],
            timestamp=timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
