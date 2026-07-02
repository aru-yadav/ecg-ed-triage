"""ECG Predictor"""

import time
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn.functional as F

from api.config import MODEL_DIR, MODEL_FILES, MODES
from api.core.safety import SafetyGuardrails
from api.ml.architecture import ImprovedECGModel

logger = logging.getLogger("ecg_api")

class ECGPredictor:
    def __init__(self):
        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.safety = SafetyGuardrails()

    def load_models(self):
        try:
            model_dir = Path(MODEL_DIR)
            for model_file in MODEL_FILES:
                model_path = model_dir / model_file
                model = ImprovedECGModel(num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                model.to(self.device)
                self.models.append(model)
            self.loaded = True
            logger.info(f"Loaded {len(self.models)} models")
        except Exception as e:
            logger.error(f"Loading failed: {e}")
            raise

    def predict(self, ecg_data, mode):
        start = time.time()
        try:
            mode = self.safety.validate_mode(mode)
            mode_config = MODES[mode]
            threshold = mode_config["threshold"]

            ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).to(self.device)

            all_probs = []
            with torch.no_grad():
                for model in self.models:
                    outputs = model(ecg_tensor)
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs[:, 1].cpu().numpy()[0])

            mi_probability = float(np.mean(all_probs))
            should_abstain, fallback_mode, confidence_score = self.safety.check_confidence(
                mi_probability, threshold, mode
            )

            if should_abstain:
                return {
                    "success": True,
                    "abstained": True,
                    "probability": mi_probability,
                    "confidence_score": confidence_score,
                    "mode": mode,
                    "mode_config": mode_config,
                    "processing_time_ms": (time.time() - start) * 1000
                }

            was_fallback = False
            if fallback_mode:
                mode = fallback_mode
                mode_config = MODES[mode]
                threshold = mode_config["threshold"]
                was_fallback = True

            prediction = "POSITIVE" if mi_probability >= threshold else "NEGATIVE"

            return {
                "success": True,
                "abstained": False,
                "prediction": prediction,
                "probability": mi_probability,
                "mode": mode,
                "mode_config": mode_config,
                "confidence_score": confidence_score,
                "was_fallback": was_fallback,
                "processing_time_ms": (time.time() - start) * 1000
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
