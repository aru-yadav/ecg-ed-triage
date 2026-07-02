"""Logging Configuration"""

import logging

def setup_logging(log_file="api_audit.log"):
    logger = logging.getLogger("ecg_api")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_prediction(logger, patient_id, prediction, probability, mode, confidence_score, was_fallback=False):
    log_entry = (
        f"AUDIT | PREDICTION | patient={patient_id} | decision={prediction} | "
        f"probability={probability:.4f} | mode={mode} | confidence={confidence_score:.4f} | "
        f"fallback={was_fallback}"
    )
    logger.info(log_entry)

def log_abstention(logger, patient_id, probability, confidence_score, reason):
    log_entry = (
        f"AUDIT | ABSTENTION | patient={patient_id} | probability={probability:.4f} | "
        f"confidence={confidence_score:.4f} | reason={reason}"
    )
    logger.warning(log_entry)
