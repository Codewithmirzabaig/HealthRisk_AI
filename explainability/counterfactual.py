"""
HealthRisk AI
Counterfactual Explanation Engine
"""

from typing import Any, Dict, List

import pandas as pd


class CounterfactualExplainer:
    """
    Generates actionable counterfactual explanations for healthcare cost predictions.
    """

    def __init__(self, model, feature_columns: List[str]):
        self.model = model
        self.feature_columns = feature_columns

    def _prepare_patient(self, patient_row: pd.DataFrame) -> pd.DataFrame:
        return patient_row.copy().reindex(columns=self.feature_columns, fill_value=0)

    def predict_cost(self, patient_row: pd.DataFrame) -> float:
        patient_row = self._prepare_patient(patient_row)
        return float(self.model.predict(patient_row)[0])

    def generate_scenarios(self, patient_row: pd.DataFrame) -> List[Dict[str, Any]]:
        patient_row = self._prepare_patient(patient_row)
        baseline_cost = self.predict_cost(patient_row)

        scenario_rules = {
            "bmi": {
                "change": -3,
                "minimum": 16,
                "label": "Reduce BMI by 3 points",
                "difficulty": "Medium",
            },
            "hospital_visits": {
                "change": -1,
                "minimum": 0,
                "label": "Reduce avoidable hospital visits by 1",
                "difficulty": "High",
            },
            "medication_count": {
                "change": -1,
                "minimum": 0,
                "label": "Reduce medication burden by 1 where clinically appropriate",
                "difficulty": "High",
            },
            "pandemic_risk_index": {
                "change": -0.10,
                "minimum": 0,
                "label": "Reduce pandemic exposure risk index by 0.10",
                "difficulty": "Low",
            },
            "claims_to_premium_ratio": {
                "change": -0.10,
                "minimum": 0,
                "label": "Improve claims-to-premium ratio by 0.10",
                "difficulty": "Medium",
            },
            "health_risk_score": {
                "change": -0.50,
                "minimum": 0,
                "label": "Improve overall health risk score by 0.50",
                "difficulty": "Medium",
            },
        }

        scenarios = []

        for feature, rule in scenario_rules.items():
            if feature not in patient_row.columns:
                continue

            modified = patient_row.copy()

            old_value = float(modified.iloc[0][feature])
            new_value = max(old_value + rule["change"], rule["minimum"])

            modified.loc[modified.index[0], feature] = new_value

            new_cost = self.predict_cost(modified)
            savings = baseline_cost - new_cost
            savings_pct = savings / baseline_cost if baseline_cost else 0

            if savings <= 0:
                confidence = 0.50
            elif savings_pct >= 0.10:
                confidence = 0.90
            elif savings_pct >= 0.05:
                confidence = 0.80
            else:
                confidence = 0.70

            scenarios.append({
                "feature": feature,
                "recommendation": rule["label"],
                "old_value": round(old_value, 3),
                "new_value": round(new_value, 3),
                "baseline_cost": round(baseline_cost, 2),
                "counterfactual_cost": round(new_cost, 2),
                "estimated_savings": round(savings, 2),
                "estimated_savings_pct": round(savings_pct * 100, 2),
                "confidence": confidence,
                "difficulty": rule["difficulty"],
            })

        scenarios = sorted(
            scenarios,
            key=lambda x: x["estimated_savings"],
            reverse=True,
        )

        return scenarios

    def explain(self, patient_row: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
        patient_row = self._prepare_patient(patient_row)
        baseline_cost = self.predict_cost(patient_row)

        recommendations = self.generate_scenarios(patient_row)[:top_n]

        if recommendations:
            best = recommendations[0]
            summary = (
                f"Baseline predicted healthcare cost is ${baseline_cost:,.2f}. "
                f"The strongest actionable counterfactual is: {best['recommendation']}. "
                f"This changes {best['feature']} from {best['old_value']} to "
                f"{best['new_value']} and may reduce cost to "
                f"${best['counterfactual_cost']:,.2f}, saving approximately "
                f"${best['estimated_savings']:,.2f} "
                f"({best['estimated_savings_pct']}%)."
            )
        else:
            summary = (
                f"Baseline predicted healthcare cost is ${baseline_cost:,.2f}. "
                "No actionable counterfactual scenarios were identified."
            )

        return {
            "baseline_prediction": round(baseline_cost, 2),
            "top_recommendations": recommendations,
            "executive_summary": summary,
        }