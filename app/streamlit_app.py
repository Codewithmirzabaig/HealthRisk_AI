import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from financial.insurance import InsuranceAnalytics
from financial.hospital_credit import HospitalCreditRisk
from financial.pharma import PharmaceuticalAnalytics
from healthcare.clinical_nlp import ClinicalNLP
from healthcare.survival_analysis import SurvivalAnalysis
from graph_ai.patient_graph import PatientGraphBuilder
from simulation.scenario_engine import HealthRiskSimulationEngine

st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="🏥",
    layout="wide",
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_financial_risk.csv"
COST_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "cost_prediction_xgb.pkl"
RISK_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "risk_classifier_xgb.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    cost_model = joblib.load(COST_MODEL_PATH)
    risk_model = joblib.load(RISK_MODEL_PATH)
    return cost_model, risk_model

df = load_data()
cost_model, risk_model = load_models()

st.sidebar.title("🏥 HealthRisk AI")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Patient Prediction",
        "Insurance Analytics",
        "Hospital Credit Risk",
        "Pharmaceutical Analytics",
        "Clinical NLP",
        "Survival Analysis",
        "Graph Analytics",
        "Simulation Lab",
        "Explainable AI",
    ],
)

st.title("HealthRisk AI Platform")

if page == "Executive Dashboard":
    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Avg Future Cost", f"${df['future_health_cost'].mean():,.0f}")
    col3.metric("High Risk Patients", f"{(df['risk_level'] == 'High').sum():,}")
    col4.metric("Avg Premium", f"${df['premium_paid'].mean():,.0f}")

    fig = px.histogram(df, x="risk_level", color="risk_level", title="Patient Risk Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="age",
        y="future_health_cost",
        color="risk_level",
        size="hospital_visits",
        title="Age vs Future Healthcare Cost",
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Patient Prediction":
    st.subheader("Live Patient Cost & Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 45)
        bmi = st.slider("BMI", 16.0, 50.0, 28.0)
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoker = st.selectbox("Smoker", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        hospital_visits = st.slider("Hospital Visits", 0, 15, 2)
        medication_count = st.slider("Medication Count", 0, 15, 3)
        pandemic_risk_index = st.slider("Pandemic Risk Index", 0.0, 1.0, 0.3)

    annual_claim_amount = (
        1000 + age * 80 + bmi * 70 + smoker * 1500
        + diabetes * 3000 + hypertension * 1800
        + heart_disease * 6000 + hospital_visits * 2200
    )

    premium_paid = (
        3000 + age * 45 + smoker * 1200
        + diabetes * 1800 + hypertension * 900
        + heart_disease * 2500
    )

    chronic_condition_count = diabetes + hypertension + heart_disease
    claims_to_premium_ratio = annual_claim_amount / premium_paid

    health_risk_score = (
        age * 0.02
        + bmi * 0.03
        + chronic_condition_count * 2
        + hospital_visits * 0.5
    )

    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "smoker": smoker,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "hospital_visits": hospital_visits,
        "medication_count": medication_count,
        "annual_claim_amount": annual_claim_amount,
        "premium_paid": premium_paid,
        "hospital_rating": 3,
        "hospital_debt_ratio": 0.45,
        "pharma_investment_score": 65,
        "esg_score": 70,
        "pandemic_risk_index": pandemic_risk_index,
        "chronic_condition_count": chronic_condition_count,
        "claims_to_premium_ratio": claims_to_premium_ratio,
        "health_risk_score": health_risk_score,
        "high_cost_patient": 0,
        "gender_Male": 1 if gender == "Male" else 0,
    }])

    input_df = input_df.reindex(columns=cost_model.feature_names_in_, fill_value=0)

    if st.button("Predict"):
        predicted_cost = cost_model.predict(input_df)[0]
        risk_pred = risk_model.predict(input_df)[0]

        risk_map = {0: "High", 1: "Low", 2: "Medium"}

        col1, col2 = st.columns(2)
        col1.metric("Predicted Future Cost", f"${predicted_cost:,.0f}")
        col2.metric("Predicted Risk Level", risk_map.get(int(risk_pred), str(risk_pred)))

        st.dataframe(input_df, use_container_width=True)

elif page == "Insurance Analytics":
    st.subheader("Insurance Actuarial Analytics")

    insurance = InsuranceAnalytics(df)
    summary = insurance.portfolio_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Loss Ratio", summary["Loss Ratio"])
    col2.metric("Combined Ratio", summary["Combined Ratio"])
    col3.metric("Avg IBNR", f"${summary['Average IBNR']:,.0f}")
    col4.metric("Avg Recommended Premium", f"${summary['Average Recommended Premium']:,.0f}")

    st.json(summary)

    fig = px.histogram(
        insurance.df,
        x="member_segment",
        color="member_segment",
        title="Member Segment Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(insurance.df.head(100), use_container_width=True)

elif page == "Hospital Credit Risk":
    st.subheader("Hospital Credit Risk Analytics")

    credit = HospitalCreditRisk(df)
    scored = credit.early_warning_flags()
    summary = credit.portfolio_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average PD", summary["avg_pd"])
    col2.metric("Avg Bond Spread BPS", summary["avg_bond_spread_bps"])
    col3.metric("Early Warning Count", summary["early_warning_count"])

    fig = px.histogram(
        scored,
        x="pd_risk_band",
        color="pd_risk_band",
        title="Probability of Default Risk Bands",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        scored[
            [
                "hospital_rating",
                "hospital_debt_ratio",
                "enhanced_credit_score",
                "probability_of_default",
                "pd_risk_band",
                "predicted_bond_spread_bps",
                "early_warning_flag",
            ]
        ].head(100),
        use_container_width=True,
    )

elif page == "Pharmaceutical Analytics":
    st.subheader("Pharmaceutical Portfolio Analytics")

    pharma = PharmaceuticalAnalytics()
    pipeline = pharma.portfolio_optimization()
    summary = pharma.portfolio_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Companies", summary["companies_analyzed"])
    col2.metric("Total rNPV", f"${summary['total_rnpv']:,.0f}M")
    col3.metric("Revenue at Risk", f"${summary['total_revenue_at_risk']:,.0f}M")

    fig = px.bar(
        pipeline,
        x="drug",
        y="portfolio_score",
        color="investment_recommendation",
        title="Drug Portfolio Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pipeline, use_container_width=True)

elif page == "Clinical NLP":
    st.subheader("Clinical NLP Note Analysis")

    nlp = ClinicalNLP()

    note = st.text_area(
        "Paste Clinical Note",
        """
Patient is a 71-year-old male with Type 2 Diabetes Mellitus,
Hypertension, Coronary Artery Disease, and Chronic Kidney Disease.
Current medications include Metformin, Lisinopril, Atorvastatin, and Aspirin.
Patient is obese, has history of smoking, and was recently readmitted.
""",
        height=200,
    )

    if st.button("Analyze Note"):
        result = nlp.analyze_note(note)
        st.json(result)

elif page == "Survival Analysis":
    st.subheader("Survival Analysis: Time to Readmission")

    survival = SurvivalAnalysis(df)

    summary = survival.risk_summary()
    st.json(summary)

    patient_id = st.number_input("Patient Row Index", min_value=0, max_value=len(df) - 1, value=0)
    days = st.slider("Prediction Horizon Days", 30, 365, 180)

    patient = df.iloc[[patient_id]]
    probability = survival.predict_survival_probability(patient, days=days)

    st.metric(f"Survival Probability at {days} Days", probability)

    km_path = PROJECT_ROOT / "reports" / "kaplan_meier_readmission.png"
    if km_path.exists():
        st.image(str(km_path), caption="Kaplan-Meier Readmission Curve", use_container_width=True)

elif page == "Graph Analytics":
    st.subheader("Patient-Disease Graph Analytics")

    graph = PatientGraphBuilder(df)
    graph.build_graph()

    st.json(graph.graph_summary())

    centrality = graph.condition_centrality()
    st.dataframe(centrality, use_container_width=True)

    fig = px.bar(
        centrality,
        x="condition",
        y="degree_centrality",
        title="Condition Centrality",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Graph-Risk Patients")
    st.dataframe(graph.top_high_risk_patients(), use_container_width=True)

elif page == "Simulation Lab":
    st.subheader("HealthRisk Lab Simulation Engine")

    years = st.slider("Simulation Years", 1, 10, 10)
    starting_capital = st.number_input("Starting Capital", value=1_000_000)

    if st.button("Run Simulation"):
        engine = HealthRiskSimulationEngine(starting_capital=starting_capital)
        history = engine.run_simulation(years=years)
        score = engine.final_score()

        st.json(score)

        fig = px.line(
            history,
            x="quarter",
            y="portfolio_value",
            title="Portfolio Value Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(history, use_container_width=True)

    st.subheader("Scenario Catalog")
    engine = HealthRiskSimulationEngine()
    st.dataframe(engine.scenario_catalog(), use_container_width=True)

elif page == "Explainable AI":
    st.subheader("Explainable AI & Governance")

    files = {
        "SHAP Feature Importance": PROJECT_ROOT / "reports" / "shap_feature_importance.png",
        "SHAP Summary": PROJECT_ROOT / "reports" / "shap_summary.png",
        "PDP Age": PROJECT_ROOT / "reports" / "pdp_age.png",
        "PDP BMI": PROJECT_ROOT / "reports" / "pdp_bmi.png",
        "ICE Age": PROJECT_ROOT / "reports" / "ice_age.png",
        "Model Card": PROJECT_ROOT / "reports" / "model_card.md",
        "Regulatory Mapping": PROJECT_ROOT / "reports" / "regulatory_mapping.md",
    }

    for name, path in files.items():
        st.markdown(f"### {name}")

        if path.exists() and path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            st.image(str(path), use_container_width=True)
        elif path.exists() and path.suffix.lower() == ".md":
            st.markdown(path.read_text(encoding="utf-8"))
        else:
            st.warning(f"Missing file: {path.name}")