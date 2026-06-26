"""
HealthRisk AI
HealthRisk Lab Simulation Engine

Implements:
- Quarterly simulation cycle
- 10 healthcare-financial scenarios
- Portfolio impact calculation
- AI recommendation logic
- 1000-point scoring framework
- Historical replay output
"""

import random
from dataclasses import dataclass, asdict
from typing import Dict, List

import pandas as pd


@dataclass
class Scenario:
    name: str
    category: str
    description: str
    insurance_impact: float
    hospital_bond_impact: float
    pharma_impact: float
    credit_facility_impact: float
    optimal_action: str


class HealthRiskSimulationEngine:
    def __init__(self, starting_capital: float = 1_000_000):
        self.starting_capital = starting_capital
        self.portfolio = {
            "insurance": starting_capital * 0.30,
            "hospital_bonds": starting_capital * 0.25,
            "pharma_equity": starting_capital * 0.30,
            "credit_facilities": starting_capital * 0.15,
        }
        self.history = []
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[Scenario]:
        return [
            Scenario(
                "Pandemic Declaration",
                "Public Health",
                "Claims surge, hospital stress, vaccine companies benefit.",
                -0.18,
                -0.12,
                0.22,
                -0.08,
                "Increase IBNR reserves and rebalance toward vaccine/diagnostic companies.",
            ),
            Scenario(
                "Hospital Staffing Shortage",
                "Operations",
                "Labor shortage increases hospital costs and reduces capacity.",
                -0.05,
                -0.14,
                0.02,
                -0.06,
                "Reduce weak hospital bond exposure and monitor operating margins.",
            ),
            Scenario(
                "Drug Recall",
                "Pharma",
                "Major drug recall hurts pharma equity and increases litigation risk.",
                -0.02,
                -0.03,
                -0.20,
                -0.04,
                "Reduce exposure to affected pharma names and diversify pipeline risk.",
            ),
            Scenario(
                "Inflation Shock",
                "Macro",
                "Medical cost inflation increases claims and pressures hospital margins.",
                -0.12,
                -0.08,
                -0.03,
                -0.05,
                "Raise premium assumptions and shorten bond duration.",
            ),
            Scenario(
                "New Breakthrough Therapy",
                "Innovation",
                "Successful therapy improves outcomes and boosts pharma valuation.",
                0.04,
                0.03,
                0.18,
                0.02,
                "Increase allocation to high-quality pharma innovators.",
            ),
            Scenario(
                "Hospital Bankruptcy",
                "Credit",
                "Regional hospital system files for bankruptcy.",
                -0.03,
                -0.22,
                -0.01,
                -0.15,
                "Exit distressed hospital bonds and tighten credit facility covenants.",
            ),
            Scenario(
                "Insurance Fraud Spike",
                "Insurance",
                "Fraudulent claims increase claim severity and reserve uncertainty.",
                -0.16,
                -0.02,
                0.00,
                -0.03,
                "Increase fraud detection controls and strengthen claims review.",
            ),
            Scenario(
                "Regulatory Reimbursement Cut",
                "Policy",
                "Government reimbursement cuts pressure hospitals and insurers.",
                -0.07,
                -0.11,
                -0.04,
                -0.05,
                "Stress test reimbursement-sensitive assets.",
            ),
            Scenario(
                "Clinical Trial Failure",
                "Pharma",
                "Late-stage trial failure reduces pharma portfolio value.",
                0.00,
                -0.01,
                -0.16,
                -0.02,
                "Reduce concentration in single-product companies.",
            ),
            Scenario(
                "Recovery Quarter",
                "Recovery",
                "Claims stabilize, hospitals recover margins, pharma rebounds.",
                0.08,
                0.10,
                0.07,
                0.04,
                "Rebalance gradually toward diversified growth exposure.",
            ),
        ]

    def run_quarter(self, quarter: int, scenario: Scenario | None = None) -> Dict:
        if scenario is None:
            scenario = random.choice(self.scenarios)

        impacts = {
            "insurance": scenario.insurance_impact,
            "hospital_bonds": scenario.hospital_bond_impact,
            "pharma_equity": scenario.pharma_impact,
            "credit_facilities": scenario.credit_facility_impact,
        }

        before_value = sum(self.portfolio.values())

        for asset, impact in impacts.items():
            self.portfolio[asset] = self.portfolio[asset] * (1 + impact)

        after_value = sum(self.portfolio.values())
        quarterly_return = (after_value - before_value) / before_value

        score = self._score_quarter(quarterly_return, scenario)

        record = {
            "quarter": quarter,
            "scenario": scenario.name,
            "category": scenario.category,
            "description": scenario.description,
            "insurance_value": round(self.portfolio["insurance"], 2),
            "hospital_bonds_value": round(self.portfolio["hospital_bonds"], 2),
            "pharma_equity_value": round(self.portfolio["pharma_equity"], 2),
            "credit_facilities_value": round(self.portfolio["credit_facilities"], 2),
            "portfolio_value": round(after_value, 2),
            "quarterly_return": round(quarterly_return, 4),
            "score": score,
            "ai_recommendation": scenario.optimal_action,
        }

        self.history.append(record)
        return record

    def _score_quarter(self, quarterly_return: float, scenario: Scenario) -> int:
        base_score = 50

        performance_points = int(quarterly_return * 500)

        risk_management_bonus = 20 if quarterly_return > -0.05 else 0

        scenario_complexity_bonus = 10 if scenario.category in [
            "Public Health",
            "Credit",
            "Pharma",
        ] else 5

        score = base_score + performance_points + risk_management_bonus + scenario_complexity_bonus

        return max(0, min(100, score))

    def run_simulation(self, years: int = 10) -> pd.DataFrame:
        total_quarters = years * 4

        for quarter in range(1, total_quarters + 1):
            self.run_quarter(quarter)

        return pd.DataFrame(self.history)

    def final_score(self) -> Dict:
        if not self.history:
            return {
                "total_score": 0,
                "final_portfolio_value": self.starting_capital,
                "certification_tier": "Not Started",
            }

        history_df = pd.DataFrame(self.history)

        raw_score = history_df["score"].sum()
        max_score = len(history_df) * 100

        normalized_score = int((raw_score / max_score) * 1000)

        final_value = history_df.iloc[-1]["portfolio_value"]

        if normalized_score >= 800:
            tier = "HealthRisk Expert Gold"
        elif normalized_score >= 650:
            tier = "HealthRisk Specialist Silver"
        elif normalized_score >= 500:
            tier = "HealthRisk Analyst Bronze"
        else:
            tier = "Training Incomplete"

        return {
            "total_score": normalized_score,
            "final_portfolio_value": round(final_value, 2),
            "certification_tier": tier,
            "quarters_completed": len(history_df),
        }

    def scenario_catalog(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(scenario) for scenario in self.scenarios])

    def save_history(self, output_path: str) -> str:
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(output_path, index=False)
        return output_path