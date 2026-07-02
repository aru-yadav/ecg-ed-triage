"""Safety Guardrails Module"""

from typing import Tuple, Optional
from api.config import CONFIDENCE_THRESHOLD, ABSTENTION_THRESHOLD

class SafetyGuardrails:

    @staticmethod
    def check_confidence(probability, threshold, mode):
        confidence_score = abs(probability - threshold)

        if confidence_score < ABSTENTION_THRESHOLD:
            return True, None, confidence_score

        if confidence_score < CONFIDENCE_THRESHOLD and mode != "max_safety":
            return False, "max_safety", confidence_score

        return False, None, confidence_score

    @staticmethod
    def classify_confidence(confidence_score):
        if confidence_score >= 0.30:
            return "HIGH"
        elif confidence_score >= CONFIDENCE_THRESHOLD:
            return "MODERATE"
        else:
            return "LOW"

    @staticmethod
    def generate_recommendation(prediction, probability, mode, confidence_level, was_abstained=False):
        if was_abstained:
            return (
                "UNABLE_TO_CLASSIFY",
                "SYSTEM UNCERTAINTY - Model confidence too low. Clinical review required.",
                "URGENT"
            )

        if prediction == "POSITIVE":
            return (
                "CRITICAL",
                f"CRITICAL: High MI risk ({probability:.1%}). Immediate cardiology consult. Confidence: {confidence_level}.",
                "URGENT"
            )
        else:
            return (
                "ROUTINE",
                f"ROUTINE: Low MI risk ({probability:.1%}). Standard protocol. Confidence: {confidence_level}.",
                "STANDARD"
            )

    @staticmethod
    def validate_mode(mode):
        from api.config import MODES
        if mode not in MODES:
            return "high_safety"
        return mode
