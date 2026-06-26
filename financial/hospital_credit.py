"""
HealthRisk AI
Hospital Credit Risk Module
"""

import pandas as pd
import numpy as np


class HospitalCreditRisk:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def traditional_scorecard(self) -> pd.DataFrame:
        self.df["traditional_credit_score"] = (
            (1 - self.df["hospital_debt_ratio"]) * 50
            + self.df["hospital_rating"] * 10
        )

        self.df["traditional_credit_rating"] = pd.cut(
            self.df["traditional_credit_score"],
            bins=[0, 40, 60, 75, 100],
            labels=["High Risk", "Moderate Risk", "Stable", "Strong"]
        )

        return self.df

    def enhanced_scorecard(self) -> pd.DataFrame:
        self.traditional_scorecard()

        self.df["clinical_quality_score"] = (
            (5 - self.df["hospital_visits"].clip(0, 5)) * 8
            + self.df["esg_score"] * 0.25
            + (1 - self.df["pandemic_risk_index"]) * 20
        )

        self.df["enhanced_credit_score"] = (
            self.df["traditional_credit_score"] * 0.55
            + self.df["clinical_quality_score"] * 0.45
        )

        self.df["enhanced_credit_rating"] = pd.cut(
            self.df["enhanced_credit_score"],
            bins=[0, 40, 60, 75, 100],
            labels=["High Risk", "Moderate Risk", "Stable", "Strong"]
        )

        return self.df

    def probability_of_default(self) -> pd.DataFrame:
        self.enhanced_scorecard()

        z = (
            -4
            + self.df["hospital_debt_ratio"] * 4
            + self.df["pandemic_risk_index"] * 2
            + self.df["hospital_visits"] * 0.15
            - self.df["hospital_rating"] * 0.35
            - self.df["esg_score"] * 0.015
        )

        self.df["probability_of_default"] = 1 / (1 + np.exp(-z))

        self.df["pd_risk_band"] = pd.cut(
            self.df["probability_of_default"],
            bins=[0, 0.05, 0.15, 0.30, 1],
            labels=["Low PD", "Moderate PD", "Elevated PD", "High PD"]
        )

        return self.df

    def bond_spread_prediction(self) -> pd.DataFrame:
        self.probability_of_default()

        self.df["predicted_bond_spread_bps"] = (
            80
            + self.df["probability_of_default"] * 600
            + self.df["hospital_debt_ratio"] * 120
            - self.df["hospital_rating"] * 10
        ).round(2)

        return self.df

    def early_warning_flags(self) -> pd.DataFrame:
        self.bond_spread_prediction()

        self.df["early_warning_flag"] = np.where(
            (
                (self.df["probability_of_default"] > 0.20)
                | (self.df["hospital_debt_ratio"] > 0.70)
                | (self.df["pandemic_risk_index"] > 0.75)
                | (self.df["hospital_rating"] <= 2)
            ),
            1,
            0
        )

        return self.df

    def portfolio_summary(self) -> dict:
        scored = self.early_warning_flags()

        return {
            "total_records": len(scored),
            "avg_pd": round(scored["probability_of_default"].mean(), 4),
            "avg_bond_spread_bps": round(scored["predicted_bond_spread_bps"].mean(), 2),
            "early_warning_count": int(scored["early_warning_flag"].sum()),
            "rating_distribution": scored["enhanced_credit_rating"].value_counts().to_dict(),
            "pd_band_distribution": scored["pd_risk_band"].value_counts().to_dict(),
        }