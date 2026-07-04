"""Main Application"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import logging

from api.config import VERSION, API_TITLE, API_DESCRIPTION, AUROC, MODES
from api.core.logging_config import setup_logging
from api.ml.predictor import ECGPredictor
from api.routers import health, prediction

logger = setup_logging()

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=VERSION)

predictor = ECGPredictor()
health.set_predictor(predictor)
prediction.set_predictor(predictor)

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])

# Web UI: serve the single-file front-end same-origin so its relative fetch()
# call to /predict works with no CORS or hardcoded host. Path is
# resolved relative to this file (like config.py) so CWD does not matter.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse(STATIC_DIR / "index.html")

@app.on_event("startup")
async def startup_event():
    predictor.load_models()
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
