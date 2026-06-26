import pandas as pd

def create_features(df):

    # Chronic conditions count
    df["chronic_condition_count"] = (
        df["diabetes"]
        + df["hypertension"]
        + df["heart_disease"]
    )

    # Claims to premium ratio
    df["claims_to_premium_ratio"] = (
        df["annual_claim_amount"]
        / df["premium_paid"]
    )

    # Health risk score
    df["health_risk_score"] = (
        df["age"] * 0.02
        + df["bmi"] * 0.03
        + df["chronic_condition_count"] * 2
        + df["hospital_visits"] * 0.5
    )

    # High-cost patient flag
    df["high_cost_patient"] = (
        df["future_health_cost"] > df["future_health_cost"].median()
    ).astype(int)

    return df