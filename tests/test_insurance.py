import pandas as pd
from financial.insurance import InsuranceAnalytics


def sample_df():
    return pd.DataFrame({
        "age": [45, 70],
        "bmi": [28, 34],
        "diabetes": [0, 1],
        "hypertension": [1, 1],
        "heart_disease": [0, 1],
        "hospital_visits": [1, 5],
        "medication_count": [2, 6],
        "annual_claim_amount": [5000, 25000],
        "premium_paid": [7000, 12000],
    })


def test_insurance_portfolio_summary():
    insurance = InsuranceAnalytics(sample_df())
    summary = insurance.portfolio_summary()

    assert "Loss Ratio" in summary
    assert "Combined Ratio" in summary
    assert "Average IBNR" in summary
    assert summary["Members"] == 2


def test_insurance_columns_created():
    insurance = InsuranceAnalytics(sample_df())
    insurance.portfolio_summary()

    expected = [
        "claims_to_premium_ratio",
        "ibnr_reserve",
        "forecast_claims",
        "recommended_premium",
        "risk_adjustment_score",
        "member_segment",
    ]

    for col in expected:
        assert col in insurance.df.columns