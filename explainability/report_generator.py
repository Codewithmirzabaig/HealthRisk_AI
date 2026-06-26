"""
HealthRisk AI
Explainability Report Generator
"""

from pathlib import Path
from datetime import datetime
import json


class ExplainabilityReport:

    def __init__(self, output_dir="reports"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.report = {
            "generated": datetime.now().isoformat(),
            "sections": {}
        }

    def add_section(self, name, content):

        self.report["sections"][name] = content

    def save_json(self, filename="explainability_report.json"):

        path = self.output_dir / filename

        with open(path, "w") as f:
            json.dump(self.report, f, indent=4)

        return path

    def summary(self):

        return {
            "generated": self.report["generated"],
            "modules": list(self.report["sections"].keys())
        }