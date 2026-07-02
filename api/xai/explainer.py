"""
ECG Explainability Module - Grad-CAM for Lead-Wise Attribution
FROZEN MODELS - READ-ONLY ACCESS
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger("ecg_api")

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

CARDIAC_REGIONS = {
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "inferior": ["II", "III", "aVF"]
}

class GradCAMExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.layer4[-1].conv2
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, ecg_tensor):
        output = self.model(ecg_tensor)
        self.model.zero_grad()
        target_score = output[:, 1]
        target_score.backward()

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(0, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(12, 1000),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def compute_lead_importance(self, cam):
        lead_activations = np.mean(cam, axis=1)
        total = np.sum(lead_activations)
        lead_percentages = (lead_activations / total) * 100
        return {LEAD_NAMES[i]: round(float(lead_percentages[i]), 2) for i in range(12)}

    def compute_cardiac_regions(self, lead_scores):
        region_scores = {}
        for region, leads in CARDIAC_REGIONS.items():
            region_scores[region] = sum(lead_scores[lead] for lead in leads)
        total = sum(region_scores.values())
        return {region: round(score / total * 100, 2) for region, score in region_scores.items()}

    def compute_temporal_importance(self, cam):
        temporal_avg = np.mean(cam, axis=0)
        segments = {
            "p_wave_region": temporal_avg[0:200].mean(),
            "qrs_region": temporal_avg[200:400].mean(),
            "st_segment_region": temporal_avg[400:700].mean(),
            "t_wave_region": temporal_avg[700:1000].mean()
        }
        total = sum(segments.values())
        return {k: round(v / total * 100, 2) for k, v in segments.items()}

    def generate_clinical_interpretation(self, lead_scores, region_scores, mi_probability):
        sorted_leads = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
        top_leads = sorted_leads[:3]
        dominant_region = max(region_scores.items(), key=lambda x: x[1])[0]

        interpretation = (
            f"The model identified ECG abnormalities with {mi_probability:.1%} MI probability. "
            f"Primary changes localized to {dominant_region} region "
            f"({region_scores[dominant_region]:.1f}% of total activation). "
        )

        interpretation += (
            f"Top contributing leads: {top_leads[0][0]} ({top_leads[0][1]:.1f}%), "
            f"{top_leads[1][0]} ({top_leads[1][1]:.1f}%), "
            f"{top_leads[2][0]} ({top_leads[2][1]:.1f}%). "
        )

        clinical_notes = {
            "anterior": "Anterior changes suggest LAD territory involvement.",
            "inferior": "Inferior changes suggest RCA or LCx territory involvement.",
            "lateral": "Lateral changes suggest LCx territory involvement.",
            "septal": "Septal changes may indicate proximal LAD involvement."
        }

        if dominant_region in clinical_notes:
            interpretation += clinical_notes[dominant_region]

        return interpretation

    def explain(self, ecg_data, mi_probability):
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).to(self.device)
        cam = self.generate_cam(ecg_tensor)

        lead_scores = self.compute_lead_importance(cam)
        region_scores = self.compute_cardiac_regions(lead_scores)
        temporal_scores = self.compute_temporal_importance(cam)

        sorted_leads = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
        top_leads = [{"lead": lead, "importance": score} for lead, score in sorted_leads[:3]]

        interpretation = self.generate_clinical_interpretation(lead_scores, region_scores, mi_probability)

        return {
            "lead_importance": lead_scores,
            "cardiac_regions": region_scores,
            "temporal_importance": temporal_scores,
            "top_contributing_leads": top_leads,
            "clinical_interpretation": interpretation,
            "explanation_method": "Grad-CAM"
        }

def create_explainer_ensemble(models, device):
    explainer = GradCAMExplainer(models[0], device)
    logger.info("Explainer initialized (read-only)")
    return explainer
