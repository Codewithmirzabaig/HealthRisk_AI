"""
HealthRisk AI
Patient-Disease Graph Module

Simplified graph analytics module for synthetic healthcare data.
Creates patient-condition graphs and graph-based risk indicators.
"""

from typing import Dict, List

import networkx as nx
import pandas as pd


class PatientGraphBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.graph = nx.Graph()

    def build_graph(self) -> nx.Graph:
        condition_columns = [
            "diabetes",
            "hypertension",
            "heart_disease",
            "smoker",
        ]

        for _, row in self.df.iterrows():
            patient_node = f"patient_{int(row['patient_id'])}"
            self.graph.add_node(patient_node, node_type="patient")

            for condition in condition_columns:
                if row[condition] == 1:
                    condition_node = f"condition_{condition}"
                    self.graph.add_node(condition_node, node_type="condition")
                    self.graph.add_edge(patient_node, condition_node)

        return self.graph

    def graph_summary(self) -> Dict:
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 6),
            "connected_components": nx.number_connected_components(self.graph),
        }

    def condition_centrality(self) -> pd.DataFrame:
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        centrality = nx.degree_centrality(self.graph)

        rows = []
        for node, value in centrality.items():
            if node.startswith("condition_"):
                rows.append({
                    "condition": node.replace("condition_", ""),
                    "degree_centrality": round(value, 6),
                })

        return pd.DataFrame(rows).sort_values(
            by="degree_centrality",
            ascending=False
        )

    def add_graph_risk_score(self) -> pd.DataFrame:
        condition_weights = {
            "diabetes": 2.0,
            "hypertension": 1.5,
            "heart_disease": 3.0,
            "smoker": 1.2,
        }

        self.df["graph_risk_score"] = 0.0

        for condition, weight in condition_weights.items():
            self.df["graph_risk_score"] += self.df[condition] * weight

        self.df["graph_risk_score"] += self.df["hospital_visits"] * 0.4
        self.df["graph_risk_score"] += self.df["medication_count"] * 0.2

        self.df["graph_risk_band"] = pd.cut(
            self.df["graph_risk_score"],
            bins=[-1, 3, 7, 100],
            labels=["Low Graph Risk", "Medium Graph Risk", "High Graph Risk"],
        )

        return self.df

    def top_high_risk_patients(self, n: int = 10) -> pd.DataFrame:
        scored = self.add_graph_risk_score()

        return scored.sort_values(
            by="graph_risk_score",
            ascending=False
        ).head(n)[
            [
                "patient_id",
                "diabetes",
                "hypertension",
                "heart_disease",
                "smoker",
                "hospital_visits",
                "medication_count",
                "graph_risk_score",
                "graph_risk_band",
            ]
        ]