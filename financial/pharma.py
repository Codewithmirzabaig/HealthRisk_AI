"""
HealthRisk AI
Pharmaceutical Analytics Module

Implements:
- Clinical trial pipeline analytics
- Phase success probability
- Patent cliff analysis
- rNPV calculation
- Portfolio ranking
"""

import numpy as np
import pandas as pd


class PharmaceuticalAnalytics:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else self.create_sample_pipeline()

    def create_sample_pipeline(self) -> pd.DataFrame:
        data = {
            "company": ["AstraNova", "BioCure", "MediGen", "OncoLife", "NeuroPath"],
            "drug": ["AN-101", "BC-220", "MG-330", "OL-440", "NP-550"],
            "therapeutic_area": ["Oncology", "Diabetes", "Cardiology", "Oncology", "Neurology"],
            "phase": ["Phase III", "Phase II", "Phase I", "Phase III", "Phase II"],
            "estimated_peak_sales": [1200, 850, 600, 1500, 900],
            "development_cost": [350, 220, 150, 420, 260],
            "years_to_launch": [2, 4, 6, 3, 5],
            "patent_years_remaining": [8, 10, 12, 5, 7],
            "trial_size": [1800, 900, 350, 2200, 750],
            "safety_score": [0.82, 0.78, 0.70, 0.75, 0.73],
            "efficacy_score": [0.88, 0.81, 0.68, 0.86, 0.76],
        }

        return pd.DataFrame(data)

    def phase_success_probability(self) -> pd.DataFrame:
        phase_base = {
            "Phase I": 0.45,
            "Phase II": 0.32,
            "Phase III": 0.58,
            "Approved": 0.95,
        }

        self.df["base_success_probability"] = self.df["phase"].map(phase_base)

        self.df["success_probability"] = (
            self.df["base_success_probability"] * 0.50
            + self.df["safety_score"] * 0.25
            + self.df["efficacy_score"] * 0.25
        ).clip(0, 1)

        return self.df

    def patent_cliff_analysis(self) -> pd.DataFrame:
        self.df["patent_cliff_risk"] = pd.cut(
            self.df["patent_years_remaining"],
            bins=[0, 5, 8, 20],
            labels=["High Patent Cliff Risk", "Moderate Patent Cliff Risk", "Low Patent Cliff Risk"],
            include_lowest=True,
        )

        self.df["revenue_at_risk"] = np.where(
            self.df["patent_years_remaining"] <= 5,
            self.df["estimated_peak_sales"] * 0.65,
            np.where(
                self.df["patent_years_remaining"] <= 8,
                self.df["estimated_peak_sales"] * 0.35,
                self.df["estimated_peak_sales"] * 0.15,
            ),
        )

        return self.df

    def calculate_rnpv(self, discount_rate: float = 0.10) -> pd.DataFrame:
        self.phase_success_probability()

        self.df["discount_factor"] = 1 / (
            (1 + discount_rate) ** self.df["years_to_launch"]
        )

        self.df["risk_adjusted_revenue"] = (
            self.df["estimated_peak_sales"]
            * self.df["success_probability"]
            * self.df["discount_factor"]
        )

        self.df["rnpv"] = (
            self.df["risk_adjusted_revenue"]
            - self.df["development_cost"]
        )

        return self.df

    def portfolio_optimization(self) -> pd.DataFrame:
        self.calculate_rnpv()
        self.patent_cliff_analysis()

        self.df["portfolio_score"] = (
            self.df["rnpv"] * 0.50
            + self.df["success_probability"] * 300
            + self.df["patent_years_remaining"] * 10
            - self.df["revenue_at_risk"] * 0.10
        )

        self.df["investment_recommendation"] = pd.cut(
            self.df["portfolio_score"],
            bins=[-10000, 100, 300, 10000],
            labels=["Avoid", "Monitor", "Invest"],
        )

        return self.df.sort_values(
            by="portfolio_score",
            ascending=False,
        )

    def portfolio_summary(self) -> dict:
        optimized = self.portfolio_optimization()

        return {
            "companies_analyzed": optimized["company"].nunique(),
            "drugs_analyzed": len(optimized),
            "avg_success_probability": round(optimized["success_probability"].mean(), 4),
            "total_rnpv": round(optimized["rnpv"].sum(), 2),
            "avg_rnpv": round(optimized["rnpv"].mean(), 2),
            "total_revenue_at_risk": round(optimized["revenue_at_risk"].sum(), 2),
            "recommendation_distribution": optimized["investment_recommendation"].value_counts().to_dict(),
        }