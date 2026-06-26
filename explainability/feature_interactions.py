"""
HealthRisk AI
Feature Interaction Analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


class FeatureInteractionAnalyzer:

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def compute_interactions(self, X: pd.DataFrame):

        interaction_values = self.explainer.shap_interaction_values(X)

        return interaction_values

    def summary_plot(
        self,
        X: pd.DataFrame,
        output_dir="reports",
        filename="shap_interaction_summary.png"
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        interaction_values = self.compute_interactions(X)

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            interaction_values,
            X,
            show=False
        )

        path = output_dir / filename

        plt.tight_layout()

        plt.savefig(path, dpi=300)

        plt.close()

        return path

    def dependence_plot(
        self,
        feature,
        X,
        output_dir="reports"
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        interaction_values = self.compute_interactions(X)

        shap.dependence_plot(
            feature,
            interaction_values,
            X,
            show=False
        )

        path = output_dir / f"interaction_{feature}.png"

        plt.tight_layout()

        plt.savefig(path, dpi=300)

        plt.close()

        return path