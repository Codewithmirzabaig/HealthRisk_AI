import pandas as pd
from graph_ai.patient_graph import PatientGraphBuilder


def sample_df():
    return pd.DataFrame({
        "patient_id": [1, 2, 3],
        "diabetes": [1, 0, 1],
        "hypertension": [1, 1, 0],
        "heart_disease": [0, 1, 1],
        "smoker": [0, 1, 0],
        "hospital_visits": [2, 4, 1],
        "medication_count": [3, 5, 2],
    })


def test_graph_builds():
    builder = PatientGraphBuilder(sample_df())
    graph = builder.build_graph()
    summary = builder.graph_summary()

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert summary["nodes"] > 0
    assert summary["edges"] > 0


def test_graph_risk_score_created():
    builder = PatientGraphBuilder(sample_df())
    scored = builder.add_graph_risk_score()

    assert "graph_risk_score" in scored.columns
    assert "graph_risk_band" in scored.columns