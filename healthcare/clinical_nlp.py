"""
HealthRisk AI
Clinical NLP Module

Simplified clinical NLP pipeline for synthetic discharge notes.

Includes:
- Text preprocessing
- Rule-based Named Entity Recognition
- Diagnosis extraction
- Medication extraction
- Risk factor extraction
- Clinical risk classification
- Structured JSON-style output

Note:
This is a lightweight substitute for ClinicalBERT/NER when real clinical
notes are unavailable. The architecture allows transformer-based models
to be added later.
"""

import re
from typing import Dict, List


class ClinicalNLP:
    def __init__(self):
        self.diagnosis_terms = {
            "diabetes": ["diabetes", "hyperglycemia", "type 2 diabetes"],
            "hypertension": ["hypertension", "high blood pressure"],
            "heart_disease": [
                "heart disease",
                "coronary artery disease",
                "cad",
                "chf",
                "congestive heart failure",
            ],
            "kidney_disease": [
                "chronic kidney disease",
                "ckd",
                "renal disease",
                "renal failure",
            ],
            "copd": [
                "copd",
                "chronic obstructive pulmonary disease",
            ],
        }

        self.medication_terms = [
            "metformin",
            "insulin",
            "lisinopril",
            "atorvastatin",
            "amlodipine",
            "warfarin",
            "aspirin",
            "albuterol",
        ]

        self.risk_terms = {
            "smoking": ["smoker", "smoking", "tobacco"],
            "obesity": ["obese", "obesity", "high bmi"],
            "readmission": ["readmission", "readmitted", "recent admission"],
            "non_adherence": [
                "non-adherent",
                "noncompliant",
                "missed medication",
                "poor adherence",
            ],
        }

    def preprocess(self, text: str) -> str:
        """
        Clean clinical text.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_diagnoses(self, text: str) -> List[str]:
        """
        Extract diagnosis concepts from clinical text.
        """
        clean_text = self.preprocess(text)
        found = []

        for diagnosis, terms in self.diagnosis_terms.items():
            if any(term in clean_text for term in terms):
                found.append(diagnosis)

        return found

    def extract_medications(self, text: str) -> List[str]:
        """
        Extract medication mentions from clinical text.
        """
        clean_text = self.preprocess(text)

        return [
            medication
            for medication in self.medication_terms
            if medication in clean_text
        ]

    def extract_risk_factors(self, text: str) -> List[str]:
        """
        Extract behavioral and utilization risk factors.
        """
        clean_text = self.preprocess(text)
        found = []

        for risk_factor, terms in self.risk_terms.items():
            if any(term in clean_text for term in terms):
                found.append(risk_factor)

        return found

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Rule-based Named Entity Recognition style extraction.
        """
        return {
            "DIAGNOSIS": self.extract_diagnoses(text),
            "MEDICATION": self.extract_medications(text),
            "RISK_FACTOR": self.extract_risk_factors(text),
        }

    def classify_risk_from_text(self, text: str) -> str:
        """
        Classify clinical note into Low, Medium, or High risk.
        """
        diagnoses = self.extract_diagnoses(text)
        medications = self.extract_medications(text)
        risks = self.extract_risk_factors(text)

        score = (
            len(diagnoses) * 2
            + len(medications) * 0.5
            + len(risks) * 2
        )

        if score >= 8:
            return "High"
        if score >= 4:
            return "Medium"
        return "Low"

    def clinical_risk_score(self, text: str) -> float:
        """
        Return numeric clinical NLP risk score.
        """
        diagnoses = self.extract_diagnoses(text)
        medications = self.extract_medications(text)
        risks = self.extract_risk_factors(text)

        score = (
            len(diagnoses) * 2
            + len(medications) * 0.5
            + len(risks) * 2
        )

        return round(score, 2)

    def analyze_note(self, text: str) -> Dict:
        """
        Full clinical note analysis.
        """
        entities = self.extract_entities(text)

        return {
            "clean_text": self.preprocess(text),
            "entities": entities,
            "diagnoses": entities["DIAGNOSIS"],
            "medications": entities["MEDICATION"],
            "risk_factors": entities["RISK_FACTOR"],
            "clinical_nlp_score": self.clinical_risk_score(text),
            "text_risk_level": self.classify_risk_from_text(text),
            "entity_count": (
                len(entities["DIAGNOSIS"])
                + len(entities["MEDICATION"])
                + len(entities["RISK_FACTOR"])
            ),
        }