"""Main Application"""

from fastapi import FastAPI
import logging

from api.config import VERSION, API_TITLE, API_DESCRIPTION, AUROC, MODES
from api.core.logging_config import setup_logging
from api.ml.predictor import ECGPredictor
from api.routers import health, prediction

# XAI imports
from api.xai.explainer import create_explainer_ensemble
from api.routers import explain

logger = setup_logging()

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=VERSION)

predictor = ECGPredictor()
health.set_predictor(predictor)
prediction.set_predictor(predictor)

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])
app.include_router(explain.router, tags=["Explainability"])

@app.on_event("startup")
async def startup_event():
    predictor.load_models()

    # Initialize explainer
    global explainer
    explainer = create_explainer_ensemble(predictor.models, predictor.device)
    explain.set_dependencies(predictor, explainer)
    logger.info("API started")

@app.get("/")
async def root():
    return {
        "service": API_TITLE,
        "version": VERSION,
        "auroc": AUROC,
        "modes": list(MODES.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
