"""Health Router"""

import time
from fastapi import APIRouter
from api.models import HealthResponse, ModeInfo
from api.config import VERSION, AUROC, MODES

router = APIRouter()
start_time = time.time()
predictor = None

def set_predictor(pred):
    global predictor
    predictor = pred

@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy" if predictor.loaded else "degraded",
        version=VERSION,
        models_loaded=len(predictor.models),
        auroc=AUROC,
        available_modes=list(MODES.keys()),
        uptime_seconds=round(uptime, 2),
        model_status="healthy" if predictor.loaded else "unhealthy"
    )

@router.get("/modes")
async def get_modes():
    return {k: ModeInfo(**v) for k, v in MODES.items()}
