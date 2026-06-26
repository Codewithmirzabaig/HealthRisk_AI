"""
HealthRisk AI
Survival Analysis Module

Implements simplified time-to-event modeling for synthetic healthcare data:
- Synthetic readmission event creation
- Kaplan-Meier survival curve
- Cox Proportional Hazards model
- Patient-level survival probability
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


class SurvivalAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cox_model = CoxPHFitter()
        self.km_model = KaplanMeierFitter()

    def create_survival_dataset(self) -> pd.DataFrame:
        """
        Create synthetic time-to-readmission data.
        """
        df = self.df.copy()

        risk_score = (
            df["age"] * 0.02
            + df["bmi"] * 0.03
            + df["diabetes"] * 1.5
            + df["hypertension"] * 1.0
            + df["heart_disease"] * 2.0
            + df["hospital_visits"] * 0.5
            + df["medication_count"] * 0.2
        )

        probability = 1 / (1 + np.exp(-(risk_score - 6)))

        df["readmission_event"] = np.random.binomial(
            1,
            probability.clip(0.05, 0.85)
        )

        df["time_to_readmission"] = np.where(
            df["readmission_event"] == 1,
            np.random.randint(7, 180, len(df)),
            np.random.randint(180, 365, len(df))
        )

        return df

    def fit_kaplan_meier(
        self,
        duration_col: str = "time_to_readmission",
        event_col: str = "readmission_event",
    ):
        survival_df = self.create_survival_dataset()

        self.km_model.fit(
            durations=survival_df[duration_col],
            event_observed=survival_df[event_col],
            label="Readmission Survival Probability"
        )

        return self.km_model

    def save_kaplan_meier_plot(
        self,
        output_dir="reports",
        filename="kaplan_meier_readmission.png"
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self.fit_kaplan_meier()

        ax = self.km_model.plot_survival_function()
        ax.set_title("Kaplan-Meier Curve: Time to Readmission")
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival Probability")

        path = output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        return path

    def fit_cox_model(self):
        survival_df = self.create_survival_dataset()

        features = [
            "age",
            "bmi",
            "diabetes",
            "hypertension",
            "heart_disease",
            "hospital_visits",
            "medication_count",
            "time_to_readmission",
            "readmission_event",
        ]

        model_df = survival_df[features].copy()

        self.cox_model.fit(
            model_df,
            duration_col="time_to_readmission",
            event_col="readmission_event"
        )

        return self.cox_model

    def cox_summary(self) -> pd.DataFrame:
        self.fit_cox_model()

        return self.cox_model.summary.reset_index()

    def predict_survival_probability(
        self,
        patient_row: pd.DataFrame,
        days: int = 180
    ) -> float:
        if not hasattr(self.cox_model, "params_"):
            self.fit_cox_model()

        patient_features = patient_row[
            [
                "age",
                "bmi",
                "diabetes",
                "hypertension",
                "heart_disease",
                "hospital_visits",
                "medication_count",
            ]
        ]

        survival_function = self.cox_model.predict_survival_function(
            patient_features
        )

        closest_day = min(
            survival_function.index,
            key=lambda x: abs(x - days)
        )

        probability = survival_function.loc[closest_day].values[0]

        return round(float(probability), 4)

    def risk_summary(self) -> Dict:
        survival_df = self.create_survival_dataset()

        return {
            "records": len(survival_df),
            "readmission_rate": round(survival_df["readmission_event"].mean(), 4),
            "avg_time_to_readmission": round(survival_df["time_to_readmission"].mean(), 2),
            "median_time_to_readmission": round(survival_df["time_to_readmission"].median(), 2),
        }