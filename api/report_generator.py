"""
ECG PDF Report Generator
Production-grade report generation for hospital deployment
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class ECGReportGenerator:
    """Generate professional PDF reports for ECG AI triage results"""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        self.images_dir = self.reports_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#DC2626'),
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#10B981'),
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6B7280'),
            alignment=TA_JUSTIFY,
            spaceBefore=20,
            borderColor=colors.HexColor('#E5E7EB'),
            borderWidth=1,
            borderPadding=10
        ))

    def _generate_ecg_image(self, ecg_data: np.ndarray, patient_id: str) -> str:
        LEAD_NAMES = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        fig, axes = plt.subplots(12, 1, figsize=(10, 12))
        fig.patch.set_facecolor('white')

        time_axis = np.arange(ecg_data.shape[1]) / 500.0

        for idx, ax in enumerate(axes):
            signal = ecg_data[idx]
            normalized = (signal - signal.mean()) / (signal.std() + 1e-8)

            ax.plot(time_axis, normalized, 'k-', linewidth=1.2)
            ax.set_ylim(-3, 3)
            ax.set_xlim(0, time_axis[-1])

            ax.grid(True, which='both', linestyle='-', linewidth=0.3, 
                   color='#FFB6C1', alpha=0.5)

            ax.text(0.02, 0.95, LEAD_NAMES[idx], transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_ylabel('mV', fontsize=8)

            if idx == 11:
                ax.set_xlabel('Time (seconds)', fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = self.images_dir / f"ecg_{patient_id}_{timestamp}.png"
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return str(image_path)

    def generate_ecg_report(
        self,
        patient_id: str,
        prediction: Dict,
        explanation: Dict,
        ecg_data: np.ndarray
    ) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{patient_id}_{timestamp}.pdf"
        report_path = self.reports_dir / report_filename

        ecg_image_path = self._generate_ecg_image(ecg_data, patient_id)

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )

        story = []

        story.append(Paragraph("🫀 ECG AI TRIAGE REPORT", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))

        patient_data = [
            ['Patient ID:', patient_id],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Analysis Mode:', prediction.get('mode', 'N/A').replace('_', ' ').title()],
            ['Model Version:', prediction.get('model_version', 'N/A')]
        ]

        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
        ]))

        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("RISK ASSESSMENT", self.styles['SectionHeader']))

        risk_category = prediction.get('risk_category', 'UNKNOWN')
        probability = prediction.get('mi_probability', 0.0) * 100
        confidence = prediction.get('confidence', 'UNKNOWN')
        priority = prediction.get('priority', 'UNKNOWN')

        risk_style = self.styles['RiskHigh'] if risk_category == 'CRITICAL' else self.styles['RiskLow']
        risk_bg = colors.HexColor('#FEE2E2') if risk_category == 'CRITICAL' else colors.HexColor('#D1FAE5')

        risk_data = [
            ['Risk Category:', Paragraph(risk_category, risk_style)],
            ['MI Probability:', f"{probability:.1f}%"],
            ['Confidence Level:', confidence],
            ['Priority:', priority]
        ]

        risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D1D5DB'))
        ]))

        story.append(risk_table)
        story.append(Spacer(1, 0.2*inch))

        recommendation = prediction.get('recommendation', 'Clinical correlation advised.')
        story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("CLINICAL INTERPRETATION", self.styles['SectionHeader']))

        clinical_text = explanation.get('clinical_interpretation', 
                                       'Clinical interpretation unavailable.')
        story.append(Paragraph(clinical_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("KEY FINDINGS", self.styles['SectionHeader']))

        top_leads = explanation.get('top_contributing_leads', [])
        if top_leads:
            leads_text = "<b>Top Contributing Leads:</b><br/>"
            for i, lead_info in enumerate(top_leads[:3], 1):
                leads_text += f"{i}. Lead {lead_info['lead']}: {lead_info['importance']:.1f}% importance<br/>"
            story.append(Paragraph(leads_text, self.styles['Normal']))

        story.append(Spacer(1, 0.1*inch))

        cardiac_regions = explanation.get('cardiac_regions', {})
        if cardiac_regions:
            regions_text = "<b>Cardiac Region Involvement:</b><br/>"
            for region, score in sorted(cardiac_regions.items(), key=lambda x: x[1], reverse=True):
                regions_text += f"• {region.capitalize()}: {score:.1f}%<br/>"
            story.append(Paragraph(regions_text, self.styles['Normal']))

        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("12-LEAD ECG WAVEFORM", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))

        ecg_img = Image(ecg_image_path, width=6.5*inch, height=7.8*inch)
        story.append(ecg_img)
        story.append(Spacer(1, 0.3*inch))

        disclaimer_text = (
            "This AI-generated report is intended for clinical decision support only "
            "and does not replace physician interpretation. Clinical correlation is required."
        )
        story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))

        story.append(Spacer(1, 0.2*inch))
        footer_text = (
            f"<font size=8 color='#6B7280'>"
            f"Generated by ECG AI Triage System v{prediction.get('model_version', '1.0')} | "
            f"Processing Time: {prediction.get('processing_time_ms', 0):.0f}ms | "
            f"Report ID: {report_filename}"
            f"</font>"
        )
        story.append(Paragraph(footer_text, self.styles['Normal']))

        doc.build(story)

        return str(report_path)


def generate_ecg_report(
    patient_id: str,
    prediction: Dict,
    explanation: Dict,
    ecg_data: np.ndarray
) -> str:
    generator = ECGReportGenerator()
    return generator.generate_ecg_report(patient_id, prediction, explanation, ecg_data)


def test_report_generation():
    print("="*80)
    print("TESTING ECG REPORT GENERATOR")
    print("="*80)

    print("\n1. Generating sample ECG data...")
    ecg_data = np.random.randn(12, 1000)
    ecg_data[1] += 0.8
    ecg_data[2] += 0.9
    ecg_data[5] += 0.7
    print("   ✓ Created 12-lead ECG (12 x 1000 samples)")

    print("\n2. Creating sample prediction...")
    prediction = {
        'patient_id': 'TEST-001',
        'risk_category': 'CRITICAL',
        'mi_probability': 0.742,
        'confidence': 'HIGH',
        'priority': 'URGENT',
        'recommendation': 'Immediate cardiology consultation and serial troponins recommended.',
        'mode': 'high_safety',
        'model_version': '1.0.0',
        'processing_time_ms': 342.5,
        'timestamp': datetime.now().isoformat()
    }
    print("   ✓ Risk: CRITICAL (74.2% probability)")

    print("\n3. Creating sample explanation...")
    explanation = {
        'clinical_interpretation': (
            'The ECG demonstrates an ST-elevation pattern predominantly in leads II, III, and aVF, '
            'which are commonly associated with the inferior myocardial region. '
            'Clinical correlation is advised.'
        ),
        'lead_importance': {
            'I': 45.2, 'II': 78.5, 'III': 82.1, 'aVR': 23.4, 'aVL': 34.2, 'aVF': 76.8,
            'V1': 28.3, 'V2': 31.5, 'V3': 29.7, 'V4': 25.8, 'V5': 22.1, 'V6': 19.4
        },
        'cardiac_regions': {
            'inferior': 79.1,
            'anterior': 28.5,
            'lateral': 35.7,
            'septal': 24.3
        },
        'top_contributing_leads': [
            {'lead': 'III', 'importance': 82.1},
            {'lead': 'II', 'importance': 78.5},
            {'lead': 'aVF', 'importance': 76.8}
        ]
    }
    print("   ✓ Primary region: Inferior (79.1%)")
    print("   ✓ Top lead: III (82.1% importance)")

    print("\n4. Generating PDF report...")
    try:
        report_path = generate_ecg_report(
            patient_id='TEST-001',
            prediction=prediction,
            explanation=explanation,
            ecg_data=ecg_data
        )

        print(f"   ✓ Report generated successfully!")
        print(f"\n{'='*80}")
        print("REPORT GENERATED:")
        print("="*80)
        print(f"📄 File: {report_path}")
        print(f"📊 Size: {os.path.getsize(report_path) / 1024:.1f} KB")
        print(f"✅ Status: Ready for clinical review")
        print("="*80)

        return report_path

    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise


if __name__ == "__main__":
    test_report_generation()
