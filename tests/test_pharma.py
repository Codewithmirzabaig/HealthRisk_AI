from financial.pharma import PharmaceuticalAnalytics


def test_pharma_pipeline_runs():
    pharma = PharmaceuticalAnalytics()

    pipeline = pharma.portfolio_optimization()
    summary = pharma.portfolio_summary()

    assert len(pipeline) >= 5
    assert "success_probability" in pipeline.columns
    assert "rnpv" in pipeline.columns
    assert "investment_recommendation" in pipeline.columns
    assert summary["drugs_analyzed"] >= 5