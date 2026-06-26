"""
HealthRisk AI
Insurance Analytics Module

Implements:
- Loss Ratio
- Combined Ratio
- IBNR Reserve Estimation
- Risk Adjustment Score
- Premium Recommendation
- Claims Forecast
"""

import pandas as pd
import numpy as np


class InsuranceAnalytics:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    ############################################################

    def add_claim_ratio(self):

        self.df["claims_to_premium_ratio"] = (
            self.df["annual_claim_amount"]
            / self.df["premium_paid"]
        )

        return self.df

    ############################################################

    def loss_ratio(self):

        total_claims = self.df["annual_claim_amount"].sum()

        total_premium = self.df["premium_paid"].sum()

        return round(total_claims / total_premium, 4)

    ############################################################

    def combined_ratio(
        self,
        expense_ratio=0.18
    ):

        lr = self.loss_ratio()

        return round(lr + expense_ratio, 4)

    ############################################################

    def estimate_ibnr(self):

        """
        Simplified IBNR estimate.

        8% of incurred claims.
        """

        self.df["ibnr_reserve"] = (
            self.df["annual_claim_amount"] * 0.08
        )

        return self.df

    ############################################################

    def risk_adjustment_score(self):

        score = (

            self.df["age"] * 0.02

            + self.df["bmi"] * 0.03

            + self.df["diabetes"] * 2

            + self.df["hypertension"] * 1.5

            + self.df["heart_disease"] * 3

            + self.df["hospital_visits"] * 0.6

            + self.df["medication_count"] * 0.2

        )

        self.df["risk_adjustment_score"] = score

        return self.df

    ############################################################

    def premium_recommendation(self):

        self.risk_adjustment_score()

        self.df["recommended_premium"] = (

            self.df["premium_paid"]

            * (

                1

                + self.df["risk_adjustment_score"] / 100

            )

        )

        return self.df

    ############################################################

    def claims_forecast(self):

        self.df["forecast_claims"] = (

            self.df["annual_claim_amount"]

            * np.random.uniform(
                1.03,
                1.08,
                len(self.df)
            )

        )

        return self.df

    ############################################################

    def member_segmentation(self):

        self.add_claim_ratio()

        self.df["member_segment"] = pd.cut(

            self.df["claims_to_premium_ratio"],

            bins=[0,0.8,1.0,1.2,100],

            labels=[

                "Highly Profitable",

                "Profitable",

                "Break-even",

                "Loss Making"

            ]

        )

        return self.df

    ############################################################

    def portfolio_summary(self):

        self.member_segmentation()

        self.estimate_ibnr()

        self.claims_forecast()

        self.premium_recommendation()

        return {

            "Members": len(self.df),

            "Loss Ratio": self.loss_ratio(),

            "Combined Ratio": self.combined_ratio(),

            "Average Premium":

                round(

                    self.df["premium_paid"].mean(),

                    2

                ),

            "Average Claims":

                round(

                    self.df["annual_claim_amount"].mean(),

                    2

                ),

            "Average IBNR":

                round(

                    self.df["ibnr_reserve"].mean(),

                    2

                ),

            "Average Forecast Claims":

                round(

                    self.df["forecast_claims"].mean(),

                    2

                ),

            "Average Recommended Premium":

                round(

                    self.df["recommended_premium"].mean(),

                    2

                ),

            "Segment Distribution":

                self.df["member_segment"]

                .value_counts()

                .to_dict()

        }