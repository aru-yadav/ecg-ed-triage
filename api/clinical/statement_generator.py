"""
Clinical Statement Generator
Rule-based, deterministic clinical interpretation for ECG patterns
Regulatory-safe (FDA CDS compliant)
"""

from typing import List, Dict, Set

LEAD_REGION_MAP = {
    "inferior": {"II", "III", "aVF"},
    "anterior": {"V2", "V3", "V4"},
    "septal": {"V1", "V2"},
    "lateral": {"I", "aVL", "V5", "V6"},
}

REGION_PHRASES = {
    "inferior": "inferior myocardial region",
    "anterior": "anterior myocardial region",
    "septal": "septal myocardial region",
    "lateral": "lateral myocardial region",
}

DEFAULT_STATEMENT = (
    "The ECG demonstrates abnormal patterns that may warrant further evaluation. "
    "Clinical correlation is advised."
)

def _detect_regions(top_leads: List[str]) -> Set[str]:
    detected = set()
    lead_set = set(top_leads)

    for region, region_leads in LEAD_REGION_MAP.items():
        overlap = lead_set & region_leads
        if len(overlap) >= 2:
            detected.add(region)

    return detected

def _format_lead_list(leads: List[str]) -> str:
    if len(leads) == 1:
        return leads[0]
    elif len(leads) == 2:
        return f"{leads[0]} and {leads[1]}"
    else:
        return ", ".join(leads[:-1]) + f", and {leads[-1]}"

def generate_clinical_statement(top_leads: List[str], cardiac_regions: Dict[str, float]) -> str:
    if not top_leads or len(top_leads) < 2:
        return DEFAULT_STATEMENT

    detected_regions = _detect_regions(top_leads)

    if not detected_regions:
        return DEFAULT_STATEMENT

    primary_region = None
    max_score = 0.0

    for region in detected_regions:
        score = cardiac_regions.get(region, 0.0)
        if score > max_score:
            max_score = score
            primary_region = region

    if not primary_region:
        return DEFAULT_STATEMENT

    region_leads = LEAD_REGION_MAP[primary_region]
    contributing_leads = [lead for lead in top_leads if lead in region_leads]

    if len(contributing_leads) < 2:
        return DEFAULT_STATEMENT

    lead_str = _format_lead_list(contributing_leads[:3])
    region_phrase = REGION_PHRASES[primary_region]

    statement = (
        f"The ECG demonstrates an ST-elevation pattern predominantly in leads {lead_str}, "
        f"which are commonly associated with the {region_phrase}. "
        f"Clinical correlation is advised."
    )

    return statement
