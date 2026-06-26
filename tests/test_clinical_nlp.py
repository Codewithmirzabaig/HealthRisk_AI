from healthcare.clinical_nlp import ClinicalNLP


def test_clinical_nlp_extracts_entities():
    nlp = ClinicalNLP()

    note = """
    Patient has type 2 diabetes, hypertension, coronary artery disease.
    Current medications include metformin, lisinopril, and aspirin.
    Patient is obese and has smoking history.
    """

    result = nlp.analyze_note(note)

    assert "diabetes" in result["diagnoses"]
    assert "hypertension" in result["diagnoses"]
    assert "heart_disease" in result["diagnoses"]
    assert "metformin" in result["medications"]
    assert "smoking" in result["risk_factors"]
    assert result["text_risk_level"] in ["Low", "Medium", "High"]


def test_clinical_nlp_low_risk_note():
    nlp = ClinicalNLP()

    note = "Patient is healthy with no major chronic disease."

    result = nlp.analyze_note(note)

    assert result["text_risk_level"] == "Low"